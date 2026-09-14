import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.cointegration.johansen import JohansenCointegration
from src.cointegration.spread_builder import SpreadBuilder
from src.execution.cost_model import CostModel
from src.execution.portfolio import PortfolioManager
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class BenchmarkRunner:

    def __init__(self, config: dict):
        self.config     = config
        self.sig_cfg    = config["signals"]
        self.cost_model = CostModel(config)
        self.portfolio  = PortfolioManager(config)
        self.metrics    = MetricsCalculator(config)

        self.static_entry_z = self.sig_cfg["entry_z"]
        self.static_exit_z  = self.sig_cfg["exit_z"]

        logger.info(
            f"BenchmarkRunner initialized | "
            f"static_entry_z={self.static_entry_z} | "
            f"static_exit_z={self.static_exit_z}"
        )

    def _run_static_ou_window(
        self,
        spread_df:    pd.DataFrame,
        funding_dict: dict,
        asset_keys:   list,
        port_state,
        train_end_idx: int
    ) -> pd.DataFrame:
     
        records = []

        for timestamp, row in spread_df.iterrows():
            zscore       = row.get("zscore", np.nan)
            spread_value = row.get("spread", np.nan)

            if np.isnan(zscore) or np.isnan(spread_value):
                continue

            date_key = timestamp.date()
            fund_b   = 0.0
            fund_o   = 0.0

            for key in asset_keys[:2]:
                if key in funding_dict:
                    df = funding_dict[key]
                    if not df.empty:
                        try:
                            daily = df["funding_rate"].resample(
                                "1D"
                            ).last()
                            ts = pd.Timestamp(date_key)
                            if ts in daily.index:
                                if key == asset_keys[0]:
                                    fund_b = float(daily.loc[ts])
                                else:
                                    fund_o = float(daily.loc[ts])
                        except Exception:
                            pass

            cost = self.cost_model.compute_round_trip_cost(
                holding_period_days  = 2.0,    # fixed assumption
                funding_rate_binance = fund_b,
                funding_rate_okx     = fund_o,
            )

            for position in list(port_state.open_positions):
                should_exit = (
                    position.direction == -1 and
                    zscore <= self.static_exit_z
                ) or (
                    position.direction == 1 and
                    zscore >= -self.static_exit_z
                )
                if should_exit:
                    self.portfolio.close_position(
                        port_state, position, timestamp,
                        spread_value, zscore, "exit_signal"
                    )

            entry_signal = (
                abs(zscore) > self.static_entry_z and
                len(port_state.open_positions) < self.config["portfolio"]["max_open_trades"]
            )

            if entry_signal:
                size = self.portfolio.compute_position_size(
                    state           = port_state,
                    zscore          = zscore,
                    entry_threshold = self.static_entry_z,
                    spread_std      = spread_df["spread"].std(),
                    is_tradeable    = True,
                )
                if size > 0:
                    self.portfolio.open_position(
                        state           = port_state,
                        timestamp       = timestamp,
                        zscore          = zscore,
                        spread_value    = spread_value,
                        size_usd        = size,
                        entry_threshold = self.static_entry_z,
                        exit_threshold  = self.static_exit_z,
                        half_life_days  = 2.0,
                        round_trip_cost = cost["round_trip_cost"],
                    )

            records.append({
                "timestamp": timestamp,
                "zscore":    zscore,
                "equity":    port_state.equity,
                "n_open":    len(port_state.open_positions),
            })

        return pd.DataFrame(records).set_index("timestamp") \
               if records else pd.DataFrame()

    def run_static_ou(
        self,
        ohlcv_dict:   dict,
        funding_dict: dict,
        asset_keys:   list
    ) -> Dict:
   
        logger.info("Running Static OU benchmark...")

        johansen    = JohansenCointegration(self.config)
        spread_bld  = SpreadBuilder(self.config)
        port_state  = self.portfolio.initialise_state()

        first_key   = asset_keys[0]
        full_index  = ohlcv_dict[first_key].index
        rows_per_day = 1440
        train_rows  = self.config["backtest"]["walk_forward_train_days"] \
                      * rows_per_day
        test_rows   = self.config["backtest"]["walk_forward_test_days"] \
                      * rows_per_day

        all_equity   = []
        train_start  = 0

        while train_start + train_rows + test_rows <= len(full_index):
            train_end  = train_start + train_rows
            test_start = train_end
            test_end   = test_start + test_rows

            train_ohlcv = {
                k: df.iloc[train_start:train_end]
                for k, df in ohlcv_dict.items()
                if k in asset_keys
            }

            is_coint, coint_vector, _ = johansen.test(
                train_ohlcv, asset_keys
            )

            if not is_coint or coint_vector is None:
                train_start += test_rows
                continue

            test_ohlcv = {
                k: df.iloc[test_start:test_end]
                for k, df in ohlcv_dict.items()
                if k in asset_keys
            }

            test_spread = spread_bld.build(
                test_ohlcv, asset_keys, coint_vector
            )

            if not test_spread.empty:
                snapshot = self._run_static_ou_window(
                    test_spread, funding_dict,
                    asset_keys, port_state, train_end
                )
                if not snapshot.empty:
                    all_equity.append(snapshot["equity"])

            train_start += test_rows

        equity_curve  = pd.concat(all_equity) \
                        if all_equity else pd.Series(dtype=float)
        trade_history = self.portfolio.get_trade_history(port_state)

        results = {
            "equity_curve":  equity_curve,
            "trade_history": trade_history,
        }

        report = self.metrics.compute(results)
        logger.info("Static OU benchmark complete")
        return report

    def run_momentum(
        self,
        ohlcv_dict:  dict,
        asset_keys:  list,
        lookback_days: int = 20
    ) -> Dict:

        logger.info("Running Momentum benchmark...")

        lookback_rows = lookback_days * 1440
        port_state    = self.portfolio.initialise_state()
        all_equity    = []

        prices = pd.DataFrame({
            k: ohlcv_dict[k]["close"]
            for k in asset_keys
            if k in ohlcv_dict
        }).dropna()

        daily_prices = prices.resample("1D").last().dropna()
        momentum = daily_prices.pct_change(lookback_days).dropna()

        equity_series = []
        capital       = self.config["portfolio"]["initial_capital"]
        current_equity = capital
        position_size  = capital * self.config["portfolio"]["max_position_pct"]

        round_trip = (
            self.cost_model.taker_fee * 4 +
            self.cost_model._slippage_cost()
        )

        for timestamp, row in momentum.iterrows():
            ranked = row.sort_values(ascending=False)

            if len(ranked) < 2:
                continue

            best  = ranked.index[0]     # long this
            worst = ranked.index[-1]    # short this

            if timestamp in daily_prices.index:
                idx = daily_prices.index.get_loc(timestamp)
                if idx + 1 < len(daily_prices):
                    next_day     = daily_prices.iloc[idx + 1]
                    best_return  = (
                        next_day[best] / daily_prices.iloc[idx][best] - 1
                    )
                    worst_return = (
                        next_day[worst] / daily_prices.iloc[idx][worst] - 1
                    )

                    gross_pnl = position_size * (
                        best_return - worst_return
                    )
                    cost       = round_trip * position_size
                    net_pnl    = gross_pnl - cost

                    current_equity += net_pnl

            equity_series.append({
                "timestamp": timestamp,
                "equity":    current_equity,
            })

        equity_df    = pd.DataFrame(equity_series).set_index("timestamp")
        equity_curve = equity_df["equity"] \
                       if not equity_df.empty else pd.Series(dtype=float)

        results = {
            "equity_curve":  equity_curve,
            "trade_history": pd.DataFrame(),
        }

        report = self.metrics.compute(results)
        logger.info("Momentum benchmark complete")
        return report

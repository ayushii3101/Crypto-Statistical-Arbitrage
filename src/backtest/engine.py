# src/backtest/engine.py

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.cointegration.johansen import JohansenCointegration
from src.cointegration.spread_builder import SpreadBuilder
from src.signals.ou_estimator import OUEstimator
from src.kalman.ou_kalman import OUKalmanFilter
from src.changepoint.bocpd import BOCPD
from src.signals.threshold import DynamicThreshold
from src.execution.cost_model import CostModel
from src.execution.portfolio import PortfolioManager, PortfolioState


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Walk-forward backtest engine supporting multiple
    simultaneous spread cases.

    Key fixes applied:
    1. Test windows use build_with_baseline() so z-scores
       are computed from training statistics, not rolling
       windows on 7 days of test data (which produces all NaN).
    2. Training spread mean/std saved and passed to test window.
    3. Complex number safety via np.real() in spread computation.

    At each window:
    - Trains models for ALL four cases
    - Trades whichever cases show cointegration
    - Respects simultaneous spread limits
    - Tracks net asset exposure across all positions
    """

    def __init__(self, config: dict):
        self.config    = config
        self.bt_cfg    = config["backtest"]
        self.mkt_cfg   = config["market_data"]
        self.coint_cfg = config["cointegration"]

        self.train_days  = self.bt_cfg["walk_forward_train_days"]
        self.test_days   = self.bt_cfg["walk_forward_test_days"]
        self.warmup      = self.bt_cfg["warmup_periods"]

        self.max_spreads = self.coint_cfg.get(
            "max_simultaneous_spreads", 2
        )

        self.primary_exchange = self.mkt_cfg["exchanges"][0]
        self.secondary_exchange = (
            self.mkt_cfg["exchanges"][1]
            if len(self.mkt_cfg["exchanges"]) > 1
            else self.mkt_cfg["exchanges"][0]
        )

        # Initialise all modules
        self.johansen   = JohansenCointegration(config)
        self.spread_bld = SpreadBuilder(config)
        self.ou_est     = OUEstimator(config)
        self.kalman     = OUKalmanFilter(config)
        self.bocpd      = BOCPD(config)
        self.threshold  = DynamicThreshold(config)
        self.cost_model = CostModel(config)
        self.portfolio  = PortfolioManager(config)

        logger.info(
            f"BacktestEngine initialized | "
            f"train={self.train_days}d | "
            f"test={self.test_days}d | "
            f"max_spreads={self.max_spreads} | "
            f"cases={list(self.coint_cfg.get('cases', {}).keys())}"
        )

    # ------------------------------------------------------------------ #
    #  Window generation                                                  #
    # ------------------------------------------------------------------ #

    def _generate_windows(
        self,
        index: pd.DatetimeIndex
    ) -> list:
        """
        Generate rolling walk-forward windows.

        Windows slide forward by test_days (not train+test).
        Train windows overlap — each uses the most recent 30 days.
        Test windows never overlap — each day is tested exactly once.
        """
        rows_per_day = 1440
        train_rows   = self.train_days * rows_per_day
        test_rows    = self.test_days  * rows_per_day
        step_rows    = test_rows
        total_rows   = len(index)
        windows      = []
        train_start  = 0

        while train_start + train_rows + test_rows <= total_rows:
            train_end  = train_start + train_rows
            test_start = train_end
            test_end   = test_start + test_rows

            windows.append((
                train_start, train_end,
                test_start,  test_end
            ))

            train_start += step_rows

        logger.info(
            f"Generated {len(windows)} walk-forward windows | "
            f"train={self.train_days}d | test={self.test_days}d"
        )
        return windows

    # ------------------------------------------------------------------ #
    #  Training phase                                                     #
    # ------------------------------------------------------------------ #

    def _train_window(
        self,
        ohlcv_dict:   dict,
        funding_dict: dict,
        train_start:  int,
        train_end:    int
    ) -> Dict[str, dict]:
        """
        Train models for ALL cases on this training window.

        For each cointegrated case, saves:
        - coint_vector:   hedge ratios for spread construction
        - spread_mean:    mean of training spread (for test z-score)
        - spread_std:     std of training spread (for test z-score)
        - current_kalman: latest Kalman state
        - bocpd_df:       changepoint probabilities

        The spread_mean and spread_std are critical — they are
        passed to build_with_baseline() in the test window so
        that z-scores are computed from training statistics,
        not from a rolling window on 7 days of test data.

        Returns empty dict if no cases are cointegrated.
        """
        # Slice training data
        train_ohlcv = {
            k: df.iloc[train_start:train_end]
            for k, df in ohlcv_dict.items()
        }

        # Test all cases simultaneously
        cointegrated = self.johansen.test_all_cases(
            train_ohlcv,
            primary_exchange=self.primary_exchange
        )

        if not cointegrated:
            logger.info(
                f"Window [{train_start}:{train_end}]: "
                f"no cases cointegrated — skipping"
            )
            return {}

        trained_cases = {}

        for case_name, case_info in cointegrated.items():

            asset_keys   = case_info["asset_keys"]
            coint_vector = case_info["coint_vector"]

            try:
                # Build training spread using rolling z-score
                # This gives us the spread statistics for baseline
                spread_df = self.spread_bld.build(
                    train_ohlcv, asset_keys, coint_vector
                )

                if spread_df.empty:
                    logger.warning(
                        f"Case {case_name}: empty training spread"
                    )
                    continue

                # Save training statistics for test window
                # These are the baseline for z-score computation
                spread_values = spread_df["spread"].dropna()
                if len(spread_values) < 100:
                    logger.warning(
                        f"Case {case_name}: insufficient spread "
                        f"values ({len(spread_values)})"
                    )
                    continue

                spread_mean = float(np.real(spread_values.mean()))
                spread_std  = float(np.real(spread_values.std()))

                if spread_std <= 0 or np.isnan(spread_std):
                    logger.warning(
                        f"Case {case_name}: invalid spread_std="
                        f"{spread_std}"
                    )
                    continue

                # OU estimation on training spread
                ou_params = self.ou_est.fit_rolling(
                    spread_values,
                    window_days=self.train_days // 2
                )

                if ou_params.empty:
                    logger.warning(
                        f"Case {case_name}: OU estimation failed"
                    )
                    continue

                # Kalman Filter on OU parameters
                kalman_df = self.kalman.filter(ou_params)

                if kalman_df.empty:
                    logger.warning(
                        f"Case {case_name}: Kalman filter failed"
                    )
                    continue

                # BOCPD structural break detection
                bocpd_df = self.bocpd.detect_on_ou_params(ou_params)
                cp_probs = self.bocpd.get_changepoint_probabilities(
                    bocpd_df
                )

                # Re-run Kalman with BOCPD reset signals
                kalman_df = self.kalman.filter(ou_params, cp_probs)

                current_kalman = self.kalman.get_current_state(
                    kalman_df
                )

                if not current_kalman.get("tradeable", False):
                    logger.info(
                        f"Case {case_name}: Kalman half-life outside "
                        f"tradeable range — skipping"
                    )
                    continue

                trained_cases[case_name] = {
                    "case_name":      case_name,
                    "asset_keys":     asset_keys,
                    "coint_vector":   coint_vector,
                    "spread_mean":    spread_mean,   # baseline for test
                    "spread_std":     spread_std,    # baseline for test
                    "kalman_df":      kalman_df,
                    "bocpd_df":       bocpd_df,
                    "current_kalman": current_kalman,
                }

                logger.info(
                    f"Case {case_name} trained ✅ | "
                    f"spread_mean={spread_mean:.4f} | "
                    f"spread_std={spread_std:.4f} | "
                    f"half_life="
                    f"{current_kalman.get('half_life_filtered',0):.2f}d"
                )

            except Exception as e:
                logger.error(
                    f"Training failed for case {case_name}: {e}",
                    exc_info=True
                )
                continue

        logger.info(
            f"Training complete | "
            f"{len(trained_cases)}/{len(cointegrated)} "
            f"cases ready to trade"
        )

        return trained_cases

    # ------------------------------------------------------------------ #
    #  Net exposure tracking                                              #
    # ------------------------------------------------------------------ #

    def _get_active_case_names(
        self,
        port_state: PortfolioState
    ) -> List[str]:
        """Get list of case names in currently open positions."""
        return [
            getattr(p, 'case_name', 'unknown')
            for p in port_state.open_positions
        ]

    def _check_exposure_ok(
        self,
        port_state:  PortfolioState,
        asset_keys:  list,
    ) -> bool:
        """
        Check that adding this position does not breach
        single-asset exposure limits.

        Prevents building up concentrated exposure to one
        asset by being long in multiple overlapping spreads.
        """
        max_exposure = self.config["portfolio"].get(
            "max_single_asset_exposure", 1.5
        )

        current_exposure = {}
        for pos in port_state.open_positions:
            pos_assets = getattr(pos, 'asset_keys', [])
            for asset in pos_assets:
                current_exposure[asset] = (
                    current_exposure.get(asset, 0) + 1.0
                )

        for asset in asset_keys:
            current = current_exposure.get(asset, 0)
            if current + 1.0 > max_exposure:
                logger.debug(
                    f"Exposure limit: {asset} at "
                    f"{current:.1f}x — blocking new position"
                )
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Testing phase                                                      #
    # ------------------------------------------------------------------ #

    def _test_window(
        self,
        ohlcv_dict:    dict,
        funding_dict:  dict,
        trained_cases: dict,
        test_start:    int,
        test_end:      int,
        port_state:    PortfolioState
    ) -> pd.DataFrame:
        """
        Generate signals for all cointegrated cases.

        KEY FIX: Uses build_with_baseline() for test spreads.

        The test window (7 days) is too short for rolling
        z-score computation (needs 30 days of warmup).
        build_with_baseline() uses training mean/std instead,
        giving valid z-scores on every single test row.

        At each minute:
        1. Process forced exits (risk management first)
        2. Process z-score exits on open positions
        3. Check entry signals for each case
        4. Record portfolio state
        """
        if not trained_cases:
            return pd.DataFrame()

        # Slice test data
        test_ohlcv = {
            k: df.iloc[test_start:test_end]
            for k, df in ohlcv_dict.items()
        }

        # Build test spreads using TRAINING statistics
        # This is the critical fix — not rolling z-score on 7 days
        case_spreads = {}
        for case_name, case_data in trained_cases.items():
            try:
                spread_df = self.spread_bld.build_with_baseline(
                    ohlcv_dict   = test_ohlcv,
                    asset_keys   = case_data["asset_keys"],
                    coint_vector = case_data["coint_vector"],
                    train_mean   = case_data["spread_mean"],
                    train_std    = case_data["spread_std"],
                )
                if not spread_df.empty:
                    case_spreads[case_name] = spread_df
                    valid = spread_df["zscore"].notna().sum()
                    logger.debug(
                        f"Test spread {case_name}: "
                        f"{len(spread_df)} rows | "
                        f"{valid} valid z-scores"
                    )
            except Exception as e:
                logger.error(
                    f"Test spread failed for {case_name}: {e}"
                )

        if not case_spreads:
            logger.warning("No valid test spreads built")
            return pd.DataFrame()

        # Get common timestamp index
        common_idx = None
        for df in case_spreads.values():
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)

        if common_idx is None or len(common_idx) == 0:
            logger.warning("Empty common index across test spreads")
            return pd.DataFrame()

        records = []

        for idx_pos, timestamp in enumerate(common_idx):

            # Skip warmup rows
            if idx_pos < self.warmup:
                continue

            bocpd_cp = self._get_bocpd_prob_at(trained_cases)

            # ── STEP 1: forced exits ───────────────────────────
            self.portfolio.check_forced_exits(
                port_state, timestamp,
                0.0, 0.0, bocpd_cp
            )

            # ── STEP 2: z-score exits on open positions ────────
            for position in list(port_state.open_positions):
                case_name = getattr(position, 'case_name', None)
                if case_name not in case_spreads:
                    continue

                row    = case_spreads[case_name].loc[timestamp]
                zscore = row.get("zscore", np.nan)
                spread = row.get("spread", np.nan)

                if np.isnan(zscore):
                    continue

                exit_z = getattr(
                    position, 'exit_threshold', 0.5
                )

                should_exit = (
                    (position.direction == -1 and
                     zscore <= exit_z) or
                    (position.direction == 1 and
                     zscore >= -exit_z)
                )

                if should_exit:
                    self.portfolio.close_position(
                        port_state, position, timestamp,
                        float(np.real(spread)),
                        zscore, "exit_signal"
                    )

            # ── STEP 3: entry signals ──────────────────────────
            active_cases = self._get_active_case_names(port_state)
            n_active     = len(set(active_cases))

            for case_name, case_data in trained_cases.items():

                if n_active >= self.max_spreads:
                    break

                if case_name in active_cases:
                    continue

                if case_name not in case_spreads:
                    continue

                row    = case_spreads[case_name].loc[timestamp]
                zscore = row.get("zscore", np.nan)
                spread = row.get("spread", np.nan)

                if np.isnan(zscore) or np.isnan(spread):
                    continue

                current_kalman = case_data["current_kalman"]
                half_life      = current_kalman.get(
                    "half_life_filtered", 2.0
                )
                spread_std     = case_data["spread_std"]

                # Funding rates
                fund_b = self._get_funding_rate(
                    funding_dict,
                    case_data["asset_keys"][0],
                    timestamp,
                    self.primary_exchange
                )
                fund_o = self._get_funding_rate(
                    funding_dict,
                    case_data["asset_keys"][0],
                    timestamp,
                    self.secondary_exchange
                )

                # Compute costs and threshold
                cost = self.cost_model.compute_round_trip_cost(
                    holding_period_days  = half_life,
                    funding_rate_binance = fund_b,
                    funding_rate_okx     = fund_o,
                )

                threshold = self.threshold.compute(
                    spread_std        = spread_std,
                    half_life_days    = half_life,
                    round_trip_cost   = cost["round_trip_cost"],
                    funding_rate_diff = cost["funding_rate_diff"],
                    bocpd_cp_prob     = bocpd_cp,
                )

                entry_z      = threshold["entry_z"]
                exit_z_new   = threshold["exit_z"]
                is_tradeable = threshold["is_tradeable"]

                if not is_tradeable:
                    continue

                if abs(zscore) <= entry_z:
                    continue

                # Check asset exposure limits
                if not self._check_exposure_ok(
                    port_state, case_data["asset_keys"]
                ):
                    continue

                # Size the position
                size = self.portfolio.compute_position_size(
                    state           = port_state,
                    zscore          = zscore,
                    entry_threshold = entry_z,
                    spread_std      = spread_std,
                    is_tradeable    = True,
                )

                if size <= 0:
                    continue

                # Open position
                position = self.portfolio.open_position(
                    state           = port_state,
                    timestamp       = timestamp,
                    zscore          = zscore,
                    spread_value    = float(np.real(spread)),
                    size_usd        = size,
                    entry_threshold = entry_z,
                    exit_threshold  = exit_z_new,
                    half_life_days  = half_life,
                    round_trip_cost = cost["round_trip_cost"],
                )

                if position is not None:
                    position.case_name  = case_name
                    position.asset_keys = case_data["asset_keys"]
                    n_active += 1

                    logger.info(
                        f"ENTRY | {case_name} | "
                        f"z={zscore:.3f} | "
                        f"threshold={entry_z:.3f} | "
                        f"size={size:.0f} | "
                        f"dir={'short' if zscore > 0 else 'long'}"
                    )

            # ── STEP 4: update PnL on open positions ──────────
            for position in port_state.open_positions:
                case_name = getattr(position, 'case_name', None)
                if case_name in case_spreads:
                    row    = case_spreads[case_name].loc[timestamp]
                    spread = float(np.real(row.get("spread", 0.0)))
                    self.portfolio.update_position_pnl(
                        position, spread
                    )

            # ── STEP 5: record state ───────────────────────────
            records.append({
                "timestamp":      timestamp,
                "equity":         port_state.equity,
                "cash":           port_state.cash,
                "n_open":         len(port_state.open_positions),
                "drawdown":       port_state.current_drawdown,
                "n_cases_active": len(
                    set(self._get_active_case_names(port_state))
                ),
                "bocpd_cp":       bocpd_cp,
            })

        return pd.DataFrame(records).set_index("timestamp") \
               if records else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _get_funding_rate(
        self,
        funding_dict:    dict,
        asset_key:       str,
        timestamp:       pd.Timestamp,
        exchange:        str
    ) -> float:
        """
        Get funding rate for an asset on a specific exchange.

        Tries to find the key for the given exchange.
        Falls back to 0.0 if not available.
        """
        # Build the key for this exchange
        # asset_key format: "binance_BTCUSDT"
        # Need: "bybit_BTCUSDT" for secondary exchange
        base_asset = "_".join(asset_key.split("_")[1:])
        key        = f"{exchange}_{base_asset}"

        if key not in funding_dict:
            return 0.0

        df = funding_dict[key]
        if df.empty or "funding_rate" not in df.columns:
            return 0.0

        try:
            daily = df["funding_rate"].resample("1D").last()
            date  = pd.Timestamp(timestamp.date(), tz="UTC")
            if date in daily.index:
                return float(daily.loc[date])
        except Exception:
            pass

        return 0.0

    def _get_bocpd_prob_at(
        self,
        trained_cases: dict,
    ) -> float:
        """
        Get maximum BOCPD changepoint probability
        across all trained cases.

        Takes the most conservative (highest) value —
        if ANY case shows a structural break, we become
        more cautious across all positions.
        """
        max_prob = 0.0

        for case_data in trained_cases.values():
            bocpd_df = case_data.get("bocpd_df", pd.DataFrame())
            if bocpd_df.empty:
                continue

            col = "combined_cp_prob"
            if col not in bocpd_df.columns:
                col = "kappa_cp_prob"
            if col not in bocpd_df.columns:
                continue

            last_prob = float(bocpd_df[col].iloc[-1])
            max_prob  = max(max_prob, last_prob)

        return max_prob

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def run(
        self,
        ohlcv_dict:   dict,
        funding_dict: dict,
    ) -> Dict:
        """
        Run the complete walk-forward backtest across all cases.

        Cases come from config["cointegration"]["cases"].
        No asset_keys parameter needed.
        """
        logger.info(
            f"Starting multi-case backtest | "
            f"cases={list(self.coint_cfg.get('cases', {}).keys())} | "
            f"max_simultaneous={self.max_spreads}"
        )

        if not ohlcv_dict:
            logger.error("Empty ohlcv_dict — cannot run backtest")
            return {}

        # Use first asset's index for window generation
        first_key  = list(ohlcv_dict.keys())[0]
        full_index = ohlcv_dict[first_key].index
        windows    = self._generate_windows(full_index)

        if not windows:
            logger.error(
                "Not enough data to generate walk-forward windows. "
                "Need at least train_days + test_days of data."
            )
            return {}

        port_state    = self.portfolio.initialise_state()
        all_equity    = []
        all_snapshots = []
        window_stats  = []

        for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):

            train_start_dt = full_index[tr_s].date()
            train_end_dt   = full_index[tr_e - 1].date()
            test_start_dt  = full_index[te_s].date()
            test_end_dt    = full_index[te_e - 1].date()

            logger.info(
                f"Window {w_idx+1}/{len(windows)} | "
                f"train [{train_start_dt} -> {train_end_dt}] | "
                f"test  [{test_start_dt} -> {test_end_dt}]"
            )

            # Training phase
            trained_cases = self._train_window(
                ohlcv_dict, funding_dict, tr_s, tr_e
            )

            n_cases       = len(trained_cases)
            equity_before = port_state.equity
            trades_before = port_state.n_trades

            # Testing phase
            snapshot = self._test_window(
                ohlcv_dict, funding_dict,
                trained_cases, te_s, te_e,
                port_state
            )

            if not snapshot.empty:
                all_snapshots.append(snapshot)
                all_equity.append(snapshot["equity"])

            window_pnl    = port_state.equity - equity_before
            window_trades = port_state.n_trades - trades_before

            window_stats.append({
                "window":          w_idx + 1,
                "train_start":     train_start_dt,
                "test_start":      test_start_dt,
                "test_end":        test_end_dt,
                "n_cases_trained": n_cases,
                "cases_traded":    list(trained_cases.keys()),
                "n_trades":        window_trades,
                "window_pnl":      round(window_pnl, 2),
                "equity_end":      round(port_state.equity, 2),
                "drawdown":        round(
                    port_state.current_drawdown, 4
                ),
            })

            logger.info(
                f"Window {w_idx+1} complete | "
                f"cases={n_cases} | "
                f"trades={window_trades} | "
                f"pnl={window_pnl:.2f} | "
                f"equity={port_state.equity:.0f}"
            )

        equity_curve  = pd.concat(all_equity) \
                        if all_equity else pd.Series(dtype=float)
        trade_history = self.portfolio.get_trade_history(port_state)
        window_df     = pd.DataFrame(window_stats)

        total_trades = len(trade_history)
        logger.info(
            f"Backtest complete | "
            f"windows={len(windows)} | "
            f"total_trades={total_trades} | "
            f"final_equity={port_state.equity:,.0f} | "
            f"total_pnl={port_state.total_pnl:,.2f}"
        )

        return {
            "equity_curve":   equity_curve,
            "trade_history":  trade_history,
            "window_stats":   window_df,
            "final_equity":   port_state.equity,
            "total_pnl":      port_state.total_pnl,
            "n_windows":      len(windows),
            "n_windows_ok":   int(
                window_df["n_cases_trained"].gt(0).sum()
            ) if not window_df.empty else 0,
        }

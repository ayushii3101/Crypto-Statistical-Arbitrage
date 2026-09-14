import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CostModel:

    def __init__(self, config: dict):
        self.config   = config
        self.cost_cfg = config["costs"]
        self.port_cfg = config["portfolio"]
        self.eval_cfg = config["evaluation"]

        self.maker_fee    = self.cost_cfg["maker_fee"]
        self.taker_fee    = self.cost_cfg["taker_fee"]
        self.slippage_bps = self.cost_cfg["slippage_bps"]
        self.funding_interval_hours = self.cost_cfg[
            "funding_rate_interval_hours"
        ]
        self.risk_free_rate = self.eval_cfg["risk_free_rate"]

        logger.info(
            f"CostModel initialized | "
            f"maker={self.maker_fee:.4f} | "
            f"taker={self.taker_fee:.4f} | "
            f"slippage={self.slippage_bps}bps"
        )

    def _exchange_fee_cost(
        self,
        use_maker: bool = False
    ) -> float:
   
        fee = self.maker_fee if use_maker else self.taker_fee
        return 4 * fee   

    def _slippage_cost(
        self,
        slippage_bps: Optional[float] = None
    ) -> float:
        
        bps = slippage_bps if slippage_bps is not None else self.slippage_bps
        return 2 * (bps / 10_000)   

    def _funding_cost(
        self,
        funding_rate_binance: float,
        funding_rate_okx: float,
        holding_period_days: float,
        position_direction: int = 1
    ) -> float:
        
        settlements_per_day = 24.0 / self.funding_interval_hours
        n_settlements = holding_period_days * settlements_per_day

        net_rate_per_settlement = position_direction * (
            funding_rate_binance - funding_rate_okx
        )

        total_funding_cost = net_rate_per_settlement * n_settlements

        logger.debug(
            f"Funding cost | "
            f"binance_rate={funding_rate_binance:.6f} | "
            f"okx_rate={funding_rate_okx:.6f} | "
            f"settlements={n_settlements:.1f} | "
            f"total={total_funding_cost:.6f}"
        )

        return total_funding_cost

    def _market_impact_cost(
        self,
        order_size_usd: float,
        adv_usd: float,
        impact_coefficient: float = 0.1
    ) -> float:
        
        if adv_usd <= 0:
            return 0.0

        participation_rate = order_size_usd / adv_usd
        impact = impact_coefficient * np.sqrt(participation_rate)

        return 2 * impact

    def _margin_opportunity_cost(
        self,
        margin_fraction: float,
        holding_period_days: float
    ) -> float:
        
        daily_rate = self.risk_free_rate / 365.0
        cost = margin_fraction * daily_rate * holding_period_days

        return cost

    def compute_round_trip_cost(
        self,
        holding_period_days: float,
        funding_rate_binance: float = 0.0,
        funding_rate_okx: float = 0.0,
        order_size_usd: float = 10_000.0,
        adv_usd: float = 1_000_000_000.0,
        use_maker: bool = False,
        position_direction: int = 1,
        margin_fraction: float = 0.1
    ) -> dict:
        
        fee_cost     = self._exchange_fee_cost(use_maker)
        slip_cost    = self._slippage_cost()
        fund_cost    = self._funding_cost(
            funding_rate_binance,
            funding_rate_okx,
            holding_period_days,
            position_direction
        )
        impact_cost  = self._market_impact_cost(order_size_usd, adv_usd)
        margin_cost  = self._margin_opportunity_cost(
            margin_fraction, holding_period_days
        )

        total = fee_cost + slip_cost + fund_cost + impact_cost + margin_cost

        result = {
            "fee_cost":          round(fee_cost, 8),
            "slippage_cost":     round(slip_cost, 8),
            "funding_cost":      round(fund_cost, 8),
            "market_impact":     round(impact_cost, 8),
            "margin_opp_cost":   round(margin_cost, 8),
            "round_trip_cost":   round(total, 8),
            "round_trip_bps":    round(total * 10_000, 4),
            "holding_days":      holding_period_days,
            "funding_rate_diff": round(
                abs(funding_rate_binance - funding_rate_okx), 8
            ),
        }

        logger.debug(
            f"Round-trip cost | "
            f"total={total:.6f} ({total*10000:.2f}bps) | "
            f"fee={fee_cost:.6f} | slip={slip_cost:.6f} | "
            f"funding={fund_cost:.6f} | "
            f"impact={impact_cost:.6f} | "
            f"margin={margin_cost:.6f}"
        )

        return result

    def compute_rolling(
        self,
        funding_binance_df: pd.DataFrame,
        funding_okx_df: pd.DataFrame,
        kalman_df: pd.DataFrame,
        order_size_usd: float = 10_000.0,
        adv_usd: float = 1_000_000_000.0
    ) -> pd.DataFrame:

        logger.info("Computing rolling cost estimates...")

        fund_b = funding_binance_df["funding_rate"].resample("1D").last()
        fund_o = funding_okx_df["funding_rate"].resample("1D").last()

        fund_aligned = pd.concat(
            [fund_b, fund_o], axis=1, join="inner"
        )
        fund_aligned.columns = ["binance", "okx"]

        if "half_life_filtered" not in kalman_df.columns:
            logger.error("half_life_filtered not in kalman_df")
            return pd.DataFrame()

        half_lives = kalman_df["half_life_filtered"]

        combined = pd.concat(
            [fund_aligned, half_lives],
            axis=1,
            join="inner"
        )
        combined.columns = ["fund_binance", "fund_okx", "half_life"]

        records = []

        for timestamp, row in combined.iterrows():

            cost = self.compute_round_trip_cost(
                holding_period_days  = row["half_life"],
                funding_rate_binance = row["fund_binance"],
                funding_rate_okx     = row["fund_okx"],
                order_size_usd       = order_size_usd,
                adv_usd              = adv_usd,
            )

            cost["timestamp"] = timestamp
            records.append(cost)

        result_df = pd.DataFrame(records).set_index("timestamp")

        logger.info(
            f"Rolling costs complete | "
            f"{len(result_df)} days | "
            f"mean_round_trip={result_df['round_trip_cost'].mean():.6f} "
            f"({result_df['round_trip_bps'].mean():.2f}bps) | "
            f"mean_funding_diff="
            f"{result_df['funding_rate_diff'].mean():.6f}"
        )

        return result_df

    def compute_funding_signal(
        self,
        funding_binance_df: pd.DataFrame,
        funding_okx_df: pd.DataFrame,
        divergence_threshold: float = 0.0001
    ) -> pd.Series:
        """
        Compute the funding rate divergence signal.

        In our system, funding rate differential serves
        a dual role:

        ROLE 1 — COST: modelled in compute_round_trip_cost()
        ROLE 2 — SIGNAL: large divergence predicts spread moves

        When Binance funding >> OKX funding:
        - Longs on Binance are paying heavily
        - Pressure to close Binance longs → BTC price falls
          on Binance relative to OKX
        - The spread between exchanges widens
        - This is a predictive signal for spread entry

        This method returns a binary signal:
            +1 when binance_funding >> okx_funding
            -1 when okx_funding >> binance_funding
             0 when rates are similar (no signal)

        The trading system uses this as an additional
        confirmation filter — only enter when both the
        z-score threshold AND the funding signal agree.

        Parameters
        ----------
        divergence_threshold : minimum |rate_diff| to generate signal
                               default = 1 basis point per 8h period
        """
        fund_b = funding_binance_df["funding_rate"].resample("1D").last()
        fund_o = funding_okx_df["funding_rate"].resample("1D").last()

        diff = fund_b - fund_o

        # Signal: +1, -1, or 0
        signal = pd.Series(0, index=diff.index, dtype=int)
        signal[diff >  divergence_threshold] =  1
        signal[diff < -divergence_threshold] = -1

        n_positive = (signal == 1).sum()
        n_negative = (signal == -1).sum()
        n_neutral  = (signal == 0).sum()

        logger.info(
            f"Funding signal computed | "
            f"positive={n_positive} | "
            f"negative={n_negative} | "
            f"neutral={n_neutral} | "
            f"active_fraction="
            f"{(n_positive + n_negative) / len(signal):.1%}"
        )

        return signal

    def summarise(
        self,
        cost_df: pd.DataFrame
    ) -> dict:
        """
        Summarise cost structure over the backtest period.

        Tells you which cost component dominates and where
        to focus if you want to reduce friction.

        In practice for crypto perp stat arb:
        - Exchange fees usually dominate (fixed, unavoidable)
        - Funding cost is highly variable (can flip to income)
        - Market impact is negligible at retail scale
        - Margin opportunity cost grows with holding period
        """
        if cost_df.empty:
            return {}

        components = [
            "fee_cost", "slippage_cost", "funding_cost",
            "market_impact", "margin_opp_cost"
        ]

        means     = cost_df[components].mean()
        dominant  = means.abs().idxmax()
        total_mean = cost_df["round_trip_cost"].mean()

        summary = {
            "mean_total_cost_bps":    round(
                cost_df["round_trip_bps"].mean(), 4
            ),
            "std_total_cost_bps":     round(
                cost_df["round_trip_bps"].std(), 4
            ),
            "dominant_component":     dominant,
            "component_breakdown":    {
                c: round(means[c] / total_mean * 100, 2)
                for c in components
                if total_mean != 0
            },
            "pct_days_funding_income": round(
                (cost_df["funding_cost"] < 0).mean() * 100, 2
            ),
            "mean_funding_rate_diff": round(
                cost_df["funding_rate_diff"].mean(), 8
            ),
        }

        logger.info(f"Cost summary: {summary}")
        return summary

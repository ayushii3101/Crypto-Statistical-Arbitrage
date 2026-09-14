import logging
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class DynamicThreshold:

    def __init__(self, config: dict):
        self.config      = config
        self.signal_cfg  = config["signals"]
        self.cost_cfg    = config["costs"]

        self.default_entry_z = self.signal_cfg["entry_z"]
        self.default_exit_z  = self.signal_cfg["exit_z"]

        self.min_half_life = self.signal_cfg["min_half_life_days"]
        self.max_half_life = self.signal_cfg["max_half_life_days"]

        logger.info(
            f"DynamicThreshold initialized | "
            f"default_entry_z={self.default_entry_z} | "
            f"default_exit_z={self.default_exit_z}"
        )

    def _compute_breakeven_z(
        self,
        spread_std: float,
        round_trip_cost: float
    ) -> float:
  
        if spread_std <= 0:
            logger.warning("spread_std <= 0, returning default entry z")
            return self.default_entry_z

        z_breakeven = round_trip_cost / spread_std

        logger.debug(f"z_breakeven={z_breakeven:.4f}")
        return z_breakeven

    def _compute_halflife_adjustment(
        self,
        half_life_days: float
    ) -> float:
        
        hl = np.clip(half_life_days, self.min_half_life, self.max_half_life)

        hl_range    = self.max_half_life - self.min_half_life
        hl_normalised = (hl - self.min_half_life) / hl_range

        adjustment = hl_normalised * 1.0

        logger.debug(
            f"half_life={half_life_days:.2f}d | "
            f"hl_adjustment={adjustment:.4f}"
        )

        return adjustment

    def _compute_funding_adjustment(
        self,
        half_life_days: float,
        funding_rate_diff: float
    ) -> float:
       
        settlements_per_day = 24 / self.cost_cfg["funding_rate_interval_hours"]
        expected_settlements = half_life_days * settlements_per_day

        funding_cost = abs(funding_rate_diff) * expected_settlements
        logger.debug(
            f"funding_rate_diff={funding_rate_diff:.6f} | "
            f"expected_settlements={expected_settlements:.1f} | "
            f"funding_cost={funding_cost:.6f}"
        )

        return funding_cost

    def _compute_safety_margin(
        self,
        half_life_days: float,
        bocpd_cp_prob: float
    ) -> float:
       
        bocpd_margin = bocpd_cp_prob * 1.0

        logger.debug(
            f"bocpd_cp_prob={bocpd_cp_prob:.3f} | "
            f"bocpd_margin={bocpd_margin:.4f}"
        )

        return bocpd_margin

    def compute(
        self,
        spread_std: float,
        half_life_days: float,
        round_trip_cost: float,
        funding_rate_diff: float = 0.0,
        bocpd_cp_prob: float = 0.0
    ) -> dict:
       
        if spread_std <= 0 or np.isnan(spread_std):
            logger.warning("Invalid spread_std — returning default thresholds")
            return {
                "entry_z":           self.default_entry_z,
                "exit_z":            self.default_exit_z,
                "z_breakeven":       self.default_entry_z,
                "hl_adjustment":     0.0,
                "funding_adjustment": 0.0,
                "safety_margin":     0.0,
                "is_tradeable":      False,
                "reason":            "invalid spread_std"
            }

        z_breakeven = self._compute_breakeven_z(
            spread_std, round_trip_cost
        )

        hl_adjustment = self._compute_halflife_adjustment(half_life_days)

        funding_cost = self._compute_funding_adjustment(
            half_life_days, funding_rate_diff
        )
        funding_z = funding_cost / spread_std if spread_std > 0 else 0.0

        safety_margin = self._compute_safety_margin(
            half_life_days, bocpd_cp_prob
        )

        entry_z = z_breakeven + hl_adjustment + funding_z + safety_margin
        entry_z = np.clip(entry_z, self.default_entry_z, 4.0)
        exit_ratio = 0.3
        exit_z = entry_z * exit_ratio
        raw_entry = z_breakeven + hl_adjustment + funding_z + safety_margin
        is_tradeable = (
            half_life_days >= self.min_half_life and
            half_life_days <= self.max_half_life and
            raw_entry <= 4.0 and
            bocpd_cp_prob < self.config["bocpd"]["threshold_probability"]
        )

        result = {
            "entry_z":            round(entry_z, 4),
            "exit_z":             round(exit_z, 4),
            "z_breakeven":        round(z_breakeven, 4),
            "hl_adjustment":      round(hl_adjustment, 4),
            "funding_adjustment": round(funding_z, 4),
            "safety_margin":      round(safety_margin, 4),
            "is_tradeable":       is_tradeable,
            "half_life_days":     round(half_life_days, 4),
            "spread_std":         round(spread_std, 6),
            "reason":             "ok" if is_tradeable else "threshold_too_high"
        }

        logger.debug(
            f"Threshold computed | "
            f"entry_z={entry_z:.4f} | exit_z={exit_z:.4f} | "
            f"breakeven={z_breakeven:.4f} | "
            f"hl_adj={hl_adjustment:.4f} | "
            f"funding_z={funding_z:.4f} | "
            f"safety={safety_margin:.4f} | "
            f"tradeable={is_tradeable}"
        )

        return result

    def compute_rolling(
        self,
        spread_df: pd.DataFrame,
        kalman_df: pd.DataFrame,
        cost_df: pd.DataFrame,
        bocpd_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        if "spread" in spread_df.columns:
            daily_spread_std = (
                spread_df["spread"]
                .resample("1D")
                .std()
                .rename("spread_std")
            )
        else:
            logger.error("spread column not found in spread_df")
            return pd.DataFrame()

        combined = pd.concat([
            daily_spread_std,
            kalman_df["half_life_filtered"],
            cost_df["round_trip_cost"],
            cost_df.get(
                "funding_rate_diff",
                pd.Series(0.0, index=cost_df.index)
            ),
            bocpd_df.get(
                "combined_cp_prob",
                pd.Series(0.0, index=bocpd_df.index)
            ).rename("cp_prob"),
        ], axis=1, join="inner")

        combined.columns = [
            "spread_std", "half_life", "round_trip_cost",
            "funding_rate_diff", "cp_prob"
        ]

        records = []

        for timestamp, row in combined.iterrows():

            threshold = self.compute(
                spread_std        = row["spread_std"],
                half_life_days    = row["half_life"],
                round_trip_cost   = row["round_trip_cost"],
                funding_rate_diff = row["funding_rate_diff"],
                bocpd_cp_prob     = row["cp_prob"],
            )

            threshold["timestamp"] = timestamp
            records.append(threshold)

        result_df = pd.DataFrame(records).set_index("timestamp")

        tradeable_pct = result_df["is_tradeable"].mean()
        logger.info(
            f"Rolling thresholds complete | "
            f"{len(result_df)} days | "
            f"tradeable={tradeable_pct:.1%} | "
            f"mean_entry_z={result_df['entry_z'].mean():.4f}"
        )

        return result_df

    def summarise(
        self,
        threshold_df: pd.DataFrame
    ) -> dict:
       
        if threshold_df.empty:
            return {}

        summary = {
            "mean_entry_z":           round(threshold_df["entry_z"].mean(), 4),
            "std_entry_z":            round(threshold_df["entry_z"].std(), 4),
            "mean_exit_z":            round(threshold_df["exit_z"].mean(), 4),
            "tradeable_fraction":     round(threshold_df["is_tradeable"].mean(), 4),
            "mean_breakeven_z":       round(threshold_df["z_breakeven"].mean(), 4),
            "mean_hl_adjustment":     round(threshold_df["hl_adjustment"].mean(), 4),
            "mean_funding_adj":       round(threshold_df["funding_adjustment"].mean(), 4),
            "mean_safety_margin":     round(threshold_df["safety_margin"].mean(), 4),
            "dominant_component":     threshold_df[[
                "z_breakeven", "hl_adjustment",
                "funding_adjustment", "safety_margin"
            ]].mean().idxmax(),
        }

        logger.info(f"Threshold summary: {summary}")
        return summary

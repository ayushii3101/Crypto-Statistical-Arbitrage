import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class OUKalmanFilter:

    def __init__(self, config: dict):
        self.config     = config
        self.kalman_cfg = config["kalman"]

        self.initial_mean = self.kalman_cfg["initial_state_mean"]
        self.initial_cov  = self.kalman_cfg["initial_state_covariance"]

        # Noise parameters — the two knobs you tune
        self.Q = self.kalman_cfg["process_noise"]       # how fast κ truly drifts
        self.R = self.kalman_cfg["observation_noise"]   # how noisy MLE κ is

        logger.info(
            f"OUKalmanFilter initialized | "
            f"Q={self.Q} | R={self.R} | "
            f"Q/R ratio={self.Q/self.R:.4f}"
        )

    def _predict(
        self,
        state_mean: float,
        state_cov: float
    ) -> Tuple[float, float]:
      
        predicted_mean = state_mean
        predicted_cov  = state_cov + self.Q

        return predicted_mean, predicted_cov

    def _update(
        self,
        predicted_mean: float,
        predicted_cov: float,
        observation: float
    ) -> Tuple[float, float, float]:
        
        innovation = observation - predicted_mean

        S             = predicted_cov + self.R
        kalman_gain   = predicted_cov / S

        updated_mean  = predicted_mean + kalman_gain * innovation
        updated_cov   = (1.0 - kalman_gain) * predicted_cov

        logger.debug(
            f"Kalman update | "
            f"obs={observation:.4f} | "
            f"pred={predicted_mean:.4f} | "
            f"innovation={innovation:.4f} | "
            f"gain={kalman_gain:.4f} | "
            f"updated={updated_mean:.4f}"
        )

        return updated_mean, updated_cov, kalman_gain

    def _apply_changepoint_reset(
        self,
        state_cov: float,
        changepoint_probability: float,
        reset_threshold: float = 0.5
    ) -> float:
       
        if changepoint_probability >= reset_threshold:
  
            blend     = changepoint_probability
            new_cov   = (
                (1.0 - blend) * state_cov +
                blend * self.initial_cov
            )
            logger.info(
                f"Changepoint reset triggered | "
                f"cp_prob={changepoint_probability:.3f} | "
                f"cov: {state_cov:.4f} → {new_cov:.4f}"
            )
            return new_cov

        return state_cov

    def filter(
        self,
        ou_params_df: pd.DataFrame,
        changepoint_probs: Optional[pd.Series] = None
    ) -> pd.DataFrame:
       
        if "kappa" not in ou_params_df.columns:
            logger.error("'kappa' column not found in OU params DataFrame")
            return pd.DataFrame()

        kappa_series = ou_params_df["kappa"].dropna()

        if len(kappa_series) < 2:
            logger.warning(
                f"Too few κ observations to filter: {len(kappa_series)}"
            )
            return pd.DataFrame()

        logger.info(
            f"Running Kalman Filter on κ series | "
            f"{len(kappa_series)} observations"
        )

        state_mean = kappa_series.iloc[0]   # start from first observation
        state_cov  = self.initial_cov

        records = []

        for timestamp, kappa_obs in kappa_series.items():

            cp_prob = 0.0
            if changepoint_probs is not None:
                if timestamp in changepoint_probs.index:
                    cp_prob = float(changepoint_probs.loc[timestamp])

            state_cov = self._apply_changepoint_reset(
                state_cov, cp_prob
            )

            pred_mean, pred_cov = self._predict(state_mean, state_cov)

            updated_mean, updated_cov, gain = self._update(
                pred_mean, pred_cov, kappa_obs
            )

            state_mean = updated_mean
            state_cov  = updated_cov

            half_life_raw      = np.log(2) / kappa_obs   if kappa_obs > 0 else np.nan
            half_life_filtered = np.log(2) / updated_mean if updated_mean > 0 else np.nan

            min_hl = self.config["signals"]["min_half_life_days"]
            max_hl = self.config["signals"]["max_half_life_days"]
            tradeable = (
                updated_mean > 0 and
                min_hl <= half_life_filtered <= max_hl
            )

            records.append({
                "timestamp":         timestamp,
                "kappa_raw":         kappa_obs,
                "kappa_filtered":    updated_mean,
                "kappa_cov":         updated_cov,
                "kalman_gain":       gain,
                "half_life_raw":     half_life_raw,
                "half_life_filtered": half_life_filtered,
                "changepoint_prob":  cp_prob,
                "tradeable":         tradeable,
            })

        result_df = pd.DataFrame(records).set_index("timestamp")

        # Summary statistics for logging
        mean_gain    = result_df["kalman_gain"].mean()
        smooth_std   = result_df["kappa_filtered"].std()
        raw_std      = result_df["kappa_raw"].std()
        noise_reduction = (1 - smooth_std / raw_std) * 100 if raw_std > 0 else 0

        logger.info(
            f"Kalman Filter complete | "
            f"mean_gain={mean_gain:.4f} | "
            f"noise_reduction={noise_reduction:.1f}% | "
            f"tradeable_days={result_df['tradeable'].sum()}"
        )

        return result_df

    def get_current_state(
        self,
        filter_results: pd.DataFrame
    ) -> dict:
        
        if filter_results.empty:
            logger.warning("Filter results empty — cannot get current state")
            return {}

        latest = filter_results.iloc[-1]

        state = {
            "kappa_filtered":     float(latest["kappa_filtered"]),
            "kappa_raw":          float(latest["kappa_raw"]),
            "kappa_uncertainty":  float(latest["kappa_cov"]),
            "half_life_filtered": float(latest["half_life_filtered"]),
            "kalman_gain":        float(latest["kalman_gain"]),
            "tradeable":          bool(latest["tradeable"]),
            "as_of":              filter_results.index[-1],
        }

        logger.info(
            f"Current Kalman state | "
            f"κ_filtered={state['kappa_filtered']:.4f} | "
            f"half_life={state['half_life_filtered']:.2f}d | "
            f"uncertainty={state['kappa_uncertainty']:.4f} | "
            f"tradeable={state['tradeable']}"
        )

        return state

    def diagnose(
        self,
        filter_results: pd.DataFrame
    ) -> dict:
       
        if filter_results.empty:
            return {"error": "empty filter results"}

        raw      = filter_results["kappa_raw"]
        filtered = filter_results["kappa_filtered"]
        gain     = filter_results["kalman_gain"]

        # Innovation series
        innovations = raw - filtered.shift(1)

        # Autocorrelation at lag 1
        innov_autocorr = innovations.autocorr(lag=1)

        noise_reduction = (
            (1 - filtered.std() / raw.std()) * 100
            if raw.std() > 0 else 0
        )

        report = {
            "n_observations":       len(filter_results),
            "noise_reduction_pct":  round(noise_reduction, 2),
            "mean_kalman_gain":     round(gain.mean(), 4),
            "std_kalman_gain":      round(gain.std(), 4),
            "innovation_autocorr":  round(innov_autocorr, 4),
            "kappa_raw_mean":       round(raw.mean(), 4),
            "kappa_filtered_mean":  round(filtered.mean(), 4),
            "kappa_raw_std":        round(raw.std(), 4),
            "kappa_filtered_std":   round(filtered.std(), 4),
            "tradeable_fraction":   round(filter_results["tradeable"].mean(), 4),
            "q_over_r_ratio":       round(self.Q / self.R, 6),
        }

        # Interpret and warn
        if noise_reduction < 20:
            logger.warning(
                "Low noise reduction (<20%) — consider reducing Q "
                "to make the filter smoother"
            )
        if noise_reduction > 70:
            logger.warning(
                "Very high noise reduction (>70%) — consider increasing Q "
                "to make the filter more responsive to real changes"
            )
        if abs(innov_autocorr) > 0.3:
            logger.warning(
                f"Innovation autocorrelation = {innov_autocorr:.3f} — "
                f"state model may be misspecified. "
                f"Consider a mean-reverting state equation instead of random walk."
            )

        logger.info(f"Kalman diagnostics: {report}")

        return report

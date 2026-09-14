# src/changepoint/bocpd.py

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


logger = logging.getLogger(__name__)


class BOCPD:
    
    def __init__(self, config: dict):
        self.config       = config
        self.bocpd_cfg    = config["bocpd"]

        self.hazard_rate  = self.bocpd_cfg["hazard_rate"]
        self.cp_threshold = self.bocpd_cfg["threshold_probability"]
        self.min_regime   = self.bocpd_cfg["min_regime_length"]

        logger.info(
            f"BOCPD initialized | "
            f"hazard={self.hazard_rate} | "
            f"expected_regime_length={1/self.hazard_rate:.0f} steps | "
            f"cp_threshold={self.cp_threshold}"
        )

    def _gaussian_log_likelihood(
        self,
        x: float,
        mean: float,
        var: float
    ) -> float:
       
        if var <= 0:
            var = 1e-8   

        return norm.logpdf(x, loc=mean, scale=np.sqrt(var))

    def _update_sufficient_statistics(
        self,
        x: float,
        run_means: np.ndarray,
        run_vars: np.ndarray,
        run_counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
       
        new_counts = run_counts + 1

        delta      = x - run_means
        new_means  = run_means + delta / new_counts
        delta2     = x - new_means
        new_vars   = (run_vars * run_counts + delta * delta2) / new_counts
        new_vars   = np.maximum(new_vars, 1e-8)    # numerical floor

        return new_means, new_vars, new_counts

    def _run_bocpd(
        self,
        series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        T         = len(series)
        cp_probs  = np.zeros(T)
        exp_runs  = np.zeros(T)

        log_R = np.array([-np.inf] * (T + 1))
        log_R[0] = 0.0    # log(1.0) = 0.0

        max_run      = T
        run_means    = np.zeros(max_run + 1)
        run_vars     = np.ones(max_run + 1)     # initialise to 1.0
        run_counts   = np.zeros(max_run + 1)

        log_hazard      = np.log(self.hazard_rate)
        log_no_hazard   = np.log(1.0 - self.hazard_rate)

        for t in range(T):
            x = series[t]

            active = np.where(log_R[: t + 1] > -np.inf)[0]

            if len(active) == 0:
                active = np.array([0])

            log_preds = np.full(t + 1, -np.inf)
            for r in active:
                log_preds[r] = self._gaussian_log_likelihood(
                    x,
                    run_means[r],
                    run_vars[r]
                )

            log_growth = log_preds[active] + log_R[active] + log_no_hazard

            log_cp_mass = (
                np.logaddexp.reduce(log_preds[active] + log_R[active])
                + log_hazard
            )

            new_log_R            = np.full(T + 1, -np.inf)
            new_log_R[0]         = log_cp_mass
            new_log_R[active + 1] = log_growth

            log_norm    = np.logaddexp.reduce(
                new_log_R[new_log_R > -np.inf]
            )
            new_log_R   = new_log_R - log_norm
            log_R = new_log_R
            cp_probs[t] = np.exp(log_R[0])

            run_lengths = np.arange(T + 1)
            probs       = np.exp(log_R)
            exp_runs[t] = np.sum(run_lengths * probs)

            run_means[1: t + 2], run_vars[1: t + 2], run_counts[1: t + 2] = \
                self._update_sufficient_statistics(
                    x,
                    run_means[: t + 1],
                    run_vars[: t + 1],
                    run_counts[: t + 1]
                )

            run_means[0]  = x
            run_vars[0]   = 1.0
            run_counts[0] = 1.0

        return cp_probs, exp_runs

    def _enforce_min_regime(
        self,
        cp_probs: np.ndarray
    ) -> np.ndarray:
       
        result           = cp_probs.copy()
        last_cp_idx      = -self.min_regime    # allows first detection

        for i in range(len(result)):
            if result[i] >= self.cp_threshold:
                if i - last_cp_idx >= self.min_regime:
                    last_cp_idx = i
                else:
                    result[i] = 0.0

        return result

    def detect(
        self,
        series: pd.Series,
        label: str = "series"
    ) -> pd.DataFrame:
        
        series_clean = series.dropna()

        if len(series_clean) < 10:
            logger.warning(
                f"BOCPD: series too short ({len(series_clean)} obs) "
                f"for meaningful detection"
            )
            return pd.DataFrame()

        logger.info(
            f"Running BOCPD on '{label}' | "
            f"{len(series_clean)} observations | "
            f"hazard={self.hazard_rate}"
        )

        cp_probs, exp_runs = self._run_bocpd(series_clean.values)
        cp_probs_filtered = self._enforce_min_regime(cp_probs)
        cp_detected = cp_probs_filtered >= self.cp_threshold
        regime_id    = np.zeros(len(series_clean), dtype=int)
        current_id   = 0
        for i in range(len(cp_detected)):
            if cp_detected[i]:
                current_id += 1
            regime_id[i] = current_id

        result = pd.DataFrame({
            "cp_probability": cp_probs_filtered,
            "cp_detected":    cp_detected,
            "expected_run":   exp_runs,
            "regime_id":      regime_id,
        }, index=series_clean.index)

        n_detected = cp_detected.sum()
        logger.info(
            f"BOCPD complete | '{label}' | "
            f"{n_detected} changepoints detected | "
            f"{current_id + 1} regimes identified"
        )

        return result

    def detect_on_ou_params(
        self,
        ou_params_df: pd.DataFrame
    ) -> pd.DataFrame:
       
        results = {}

        if "kappa" in ou_params_df.columns:
            kappa_result = self.detect(
                ou_params_df["kappa"],
                label="kappa"
            )
            if not kappa_result.empty:
                results["kappa"] = kappa_result

        if "sigma" in ou_params_df.columns:
            sigma_result = self.detect(
                ou_params_df["sigma"],
                label="sigma"
            )
            if not sigma_result.empty:
                results["sigma"] = sigma_result

        if not results:
            logger.error("BOCPD: no valid series to process")
            return pd.DataFrame()

        combined = pd.DataFrame(index=ou_params_df.dropna().index)

        if "kappa" in results:
            combined["kappa_cp_prob"]     = results["kappa"]["cp_probability"]
            combined["kappa_cp_detected"] = results["kappa"]["cp_detected"]
            combined["kappa_regime_id"]   = results["kappa"]["regime_id"]

        if "sigma" in results:
            combined["sigma_cp_prob"]     = results["sigma"]["cp_probability"]
            combined["sigma_cp_detected"] = results["sigma"]["cp_detected"]

        prob_cols = [c for c in combined.columns if c.endswith("_cp_prob")]
        if prob_cols:
            combined["combined_cp_prob"] = combined[prob_cols].max(axis=1)
            combined["combined_cp_detected"] = (
                combined["combined_cp_prob"] >= self.cp_threshold
            )

        n_combined = combined.get(
            "combined_cp_detected",
            pd.Series(False, index=combined.index)
        ).sum()

        logger.info(
            f"Combined BOCPD result | "
            f"{n_combined} combined changepoints detected"
        )

        return combined

    def get_changepoint_probabilities(
        self,
        bocpd_results: pd.DataFrame
    ) -> pd.Series:
        
        if "combined_cp_prob" in bocpd_results.columns:
            return bocpd_results["combined_cp_prob"]

        if "kappa_cp_prob" in bocpd_results.columns:
            return bocpd_results["kappa_cp_prob"]

        logger.warning(
            "No changepoint probability column found in BOCPD results"
        )
        return pd.Series(dtype=float)

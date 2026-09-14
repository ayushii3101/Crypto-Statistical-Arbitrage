import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


class PermutationTest:

    def __init__(self, config: dict):
        self.config   = config
        self.eval_cfg = config["evaluation"]
        self.n_trials = self.eval_cfg["permutation_trials"]
        self.seed     = self.config["project"]["random_seed"]
        self.rf_rate  = self.eval_cfg["risk_free_rate"]

        logger.info(
            f"PermutationTest initialized | "
            f"n_trials={self.n_trials} | seed={self.seed}"
        )

    def _compute_sharpe(
        self,
        returns: np.ndarray,
        periods_per_year: int
    ) -> float:
   
        std = returns.std()
        if std == 0:
            return 0.0

        rf_per_period = self.rf_rate / periods_per_year
        excess        = returns - rf_per_period
        sharpe        = (excess.mean() / excess.std()) * \
                        np.sqrt(periods_per_year)
        return float(sharpe)

    def _compute_calmar(
        self,
        returns: np.ndarray,
        periods_per_year: int
    ) -> float:
     
        equity = np.exp(np.cumsum(returns))
        hwm    = np.maximum.accumulate(equity)
        dd     = (hwm - equity) / hwm
        max_dd = dd.max()

        if max_dd <= 0:
            return np.inf

        annual_return = returns.mean() * periods_per_year
        return float(annual_return / max_dd)

    def run(
        self,
        equity_curve: pd.Series,
        test_metric:  str = "sharpe"
    ) -> Dict:

        if equity_curve.empty or len(equity_curve) < 100:
            logger.error(
                "Equity curve too short for permutation test"
            )
            return {}

        log_returns = np.log(
            equity_curve / equity_curve.shift(1)
        ).dropna().values

        periods_per_year = 252 * 1440    # minute-level

        if test_metric == "calmar":
            observed = self._compute_calmar(
                log_returns, periods_per_year
            )
        else:
            observed = self._compute_sharpe(
                log_returns, periods_per_year
            )

        logger.info(
            f"Permutation test | "
            f"metric={test_metric} | "
            f"observed_{test_metric}={observed:.4f} | "
            f"running {self.n_trials} permutations..."
        )

        rng = np.random.default_rng(self.seed)

        null_distribution = np.zeros(self.n_trials)

        for i in range(self.n_trials):
            # Shuffle return order — destroys temporal structure
            shuffled = rng.permutation(log_returns)

            if test_metric == "calmar":
                null_distribution[i] = self._compute_calmar(
                    shuffled, periods_per_year
                )
            else:
                null_distribution[i] = self._compute_sharpe(
                    shuffled, periods_per_year
                )

            if (i + 1) % 100 == 0:
                logger.debug(
                    f"Permutation {i+1}/{self.n_trials} complete"
                )

        p_value = float(
            np.mean(null_distribution >= observed)
        )

        percentile_rank = float(
            stats.percentileofscore(null_distribution, observed)
        )

        null_mean = null_distribution.mean()
        null_std  = null_distribution.std()
        z_score   = float(
            (observed - null_mean) / null_std
            if null_std > 0 else 0.0
        )

        is_significant = p_value < 0.05

        result = {
            "metric":           test_metric,
            "observed_stat":    round(observed, 4),
            "p_value":          round(p_value, 6),
            "is_significant":   is_significant,
            "percentile_rank":  round(percentile_rank, 2),
            "z_score":          round(z_score, 4),
            "null_mean":        round(null_mean, 4),
            "null_std":         round(null_std, 4),
            "null_95th_pctile": round(
                np.percentile(null_distribution, 95), 4
            ),
            "n_trials":         self.n_trials,
            "null_distribution": null_distribution,
        }

        self._log_result(result)
        return result

    def run_both_metrics(
        self,
        equity_curve: pd.Series
    ) -> Dict:
       
        logger.info("Running permutation test on Sharpe and Calmar...")

        sharpe_result = self.run(equity_curve, test_metric="sharpe")
        calmar_result = self.run(equity_curve, test_metric="calmar")

        combined = {
            "sharpe": sharpe_result,
            "calmar": calmar_result,
            "both_significant": (
                sharpe_result.get("is_significant", False) and
                calmar_result.get("is_significant", False)
            ),
            "interpretation": self._interpret(
                sharpe_result, calmar_result
            ),
        }

        return combined

    def _interpret(
        self,
        sharpe_result: Dict,
        calmar_result: Dict
    ) -> str:
       
        s_sig = sharpe_result.get("is_significant", False)
        c_sig = calmar_result.get("is_significant", False)
        s_p   = sharpe_result.get("p_value", 1.0)
        c_p   = calmar_result.get("p_value", 1.0)

        if s_sig and c_sig:
            return (
                f"Strong evidence of genuine alpha. "
                f"Both Sharpe (p={s_p:.3f}) and Calmar (p={c_p:.3f}) "
                f"are statistically significant. The strategy's temporal "
                f"return structure is unlikely to arise by chance."
            )
        elif s_sig:
            return (
                f"Moderate evidence of alpha. Sharpe is significant "
                f"(p={s_p:.3f}) but Calmar is not (p={c_p:.3f}). "
                f"Average returns appear genuine but drawdown control "
                f"may be sample-dependent."
            )
        elif c_sig:
            return (
                f"Partial evidence of alpha. Calmar is significant "
                f"(p={c_p:.3f}) but Sharpe is not (p={s_p:.3f}). "
                f"Risk management appears genuine but average returns "
                f"may be lucky."
            )
        else:
            return (
                f"No statistically significant alpha detected. "
                f"Sharpe p={s_p:.3f}, Calmar p={c_p:.3f}. "
                f"Strategy performance is not distinguishable "
                f"from randomly ordered returns."
            )

    def _log_result(self, result: Dict) -> None:
        sig_str = "SIGNIFICANT" if result["is_significant"] else "NOT SIGNIFICANT"
        logger.info(
            f"Permutation Test Result | "
            f"{result['metric'].upper()} | {sig_str} | "
            f"observed={result['observed_stat']:.4f} | "
            f"p={result['p_value']:.4f} | "
            f"percentile={result['percentile_rank']:.1f} | "
            f"z={result['z_score']:.3f}"
        )

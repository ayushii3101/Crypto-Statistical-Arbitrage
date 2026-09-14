import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


logger = logging.getLogger(__name__)


class OUEstimator:

    def __init__(self, config: dict):
        self.config      = config
        self.signal_cfg  = config["signals"]
        self.backtest_cfg = config["backtest"]

        self.min_half_life = self.signal_cfg["min_half_life_days"]
        self.max_half_life = self.signal_cfg["max_half_life_days"]

        logger.info(
            f"OUEstimator initialized | "
            f"half_life_range=[{self.min_half_life}, "
            f"{self.max_half_life}] days"
        )

    def _ols_initialise(
        self,
        spread: pd.Series
    ) -> Tuple[float, float, float]:

        X_lag = spread.shift(1).dropna()
        X_cur = spread[1:]

        X_lag, X_cur = X_lag.align(X_cur, join="inner")

        n   = len(X_cur)
        dt  = 1.0 / 1440.0    # one minute in day units

        cov_matrix = np.cov(X_cur.values, X_lag.values)
        beta  = cov_matrix[0, 1] / cov_matrix[1, 1]
        alpha = X_cur.mean() - beta * X_lag.mean()

        if beta >= 1.0:
            logger.warning(
                f"OLS: β={beta:.4f} >= 1.0 — "
                f"spread may not be stationary. "
                f"Clamping to 0.99 for initialisation."
            )
            beta = 0.99

        if beta <= 0.0:
            logger.warning(
                f"OLS: β={beta:.4f} <= 0.0 — "
                f"clamping to 0.01 for initialisation."
            )
            beta = 0.01

        kappa = -np.log(beta) / dt
        mu    = alpha / (1.0 - beta)

        residuals = X_cur.values - (alpha + beta * X_lag.values)
        sigma     = np.std(residuals) / np.sqrt(dt)

        logger.debug(
            f"OLS initialisation | "
            f"κ={kappa:.4f} | μ={mu:.6f} | σ={sigma:.6f}"
        )

        return kappa, mu, sigma

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        spread: np.ndarray,
        dt: float
    ) -> float:
       
        kappa, mu, sigma = params

        if kappa <= 0 or sigma <= 0:
            return 1e10

        e_kdt       = np.exp(-kappa * dt)
        one_minus_e = 1.0 - e_kdt

        mean_cond = spread[:-1] * e_kdt + mu * one_minus_e
        var_cond  = (sigma ** 2) * (1.0 - np.exp(-2 * kappa * dt)) / (2 * kappa)

        if var_cond <= 0:
            return 1e10

        log_likelihood = np.sum(
            norm.logpdf(spread[1:], loc=mean_cond, scale=np.sqrt(var_cond))
        )

        return -log_likelihood

    def _compute_half_life(self, kappa: float) -> float:
        
        if kappa <= 0:
            return np.inf

        return np.log(2.0) / kappa

    def _is_tradeable(
        self,
        kappa: float,
        half_life: float
    ) -> Tuple[bool, str]:
       
        if kappa <= 0:
            return False, f"κ={kappa:.4f} <= 0: spread is not mean-reverting"

        if half_life < self.min_half_life:
            return (
                False,
                f"half_life={half_life:.2f}d < "
                f"min={self.min_half_life}d: reverts too fast"
            )

        if half_life > self.max_half_life:
            return (
                False,
                f"half_life={half_life:.2f}d > "
                f"max={self.max_half_life}d: reverts too slowly"
            )

        return True, f"Tradeable: half_life={half_life:.2f}d"

    def fit(
        self,
        spread: pd.Series
    ) -> Optional[dict]:
       
        spread = spread.dropna()

        if len(spread) < 50:
            logger.warning(
                f"Spread too short for OU fitting: {len(spread)} rows"
            )
            return None

        dt = 1.0 / 1440.0    # one minute in day units

        try:
            kappa_init, mu_init, sigma_init = self._ols_initialise(spread)
        except Exception as e:
            logger.error(f"OLS initialisation failed: {e}")
            return None

        initial_params = [kappa_init, mu_init, sigma_init]
        bounds = [
            (1e-6, None),     # kappa > 0
            (None, None),     # mu unconstrained
            (1e-6, None),     # sigma > 0
        ]

        try:
            result = minimize(
                self._negative_log_likelihood,
                x0=initial_params,
                args=(spread.values, dt),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 1000,
                    "ftol":    1e-12,   # tight tolerance for precision
                }
            )

            if not result.success:
                logger.warning(
                    f"MLE optimisation did not converge: {result.message}. "
                    f"Falling back to OLS estimates."
                )
                kappa, mu, sigma = kappa_init, mu_init, sigma_init
            else:
                kappa, mu, sigma = result.x

        except Exception as e:
            logger.error(f"MLE optimisation failed: {e}")
            kappa, mu, sigma = kappa_init, mu_init, sigma_init

        half_life             = self._compute_half_life(kappa)
        tradeable, reason     = self._is_tradeable(kappa, half_life)

        params = {
            "kappa":     float(kappa),
            "mu":        float(mu),
            "sigma":     float(sigma),
            "half_life": float(half_life),
            "tradeable": tradeable,
            "reason":    reason,
            "n_obs":     len(spread),
        }

        logger.info(
            f"OU fit complete | "
            f"κ={kappa:.4f} | μ={mu:.6f} | σ={sigma:.6f} | "
            f"half_life={half_life:.2f}d | tradeable={tradeable}"
        )

        return params

    def fit_rolling(
        self,
        spread: pd.Series,
        window_days: int = 30
    ) -> pd.DataFrame:
  
        window_rows = window_days * 1440
        step_rows   = 1440    # re-estimate once per day

        records = []
        spread_values = spread.dropna()

        logger.info(
            f"Rolling OU estimation | "
            f"window={window_days}d | "
            f"total_rows={len(spread_values)}"
        )

        for end_idx in range(window_rows, len(spread_values), step_rows):

            start_idx   = end_idx - window_rows
            window      = spread_values.iloc[start_idx:end_idx]
            window_end  = spread_values.index[end_idx - 1]

            params = self.fit(window)

            if params is None:
                records.append({
                    "timestamp": window_end,
                    "kappa":     np.nan,
                    "mu":        np.nan,
                    "sigma":     np.nan,
                    "half_life": np.nan,
                    "tradeable": False,
                })
            else:
                records.append({
                    "timestamp": window_end,
                    **{k: params[k] for k in
                       ["kappa", "mu", "sigma", "half_life", "tradeable"]}
                })

        df = pd.DataFrame(records).set_index("timestamp")

        tradeable_rate = df["tradeable"].mean()
        logger.info(
            f"Rolling OU complete | "
            f"{len(df)} windows | "
            f"tradeable_rate={tradeable_rate:.1%}"
        )

        return df

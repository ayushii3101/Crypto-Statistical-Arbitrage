import logging
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


class JohansenCointegration:
    """
    Tests cointegration for multiple asset cases simultaneously.

    Supports four cases:
        BTC-ETH, BTC-SOL, ETH-SOL, BTC-ETH-SOL

    Each case is tested independently with its own
    cointegrating vector. The engine trades whichever
    cases show cointegration in each window.

    Key fixes applied:
    1. np.real() strips imaginary parts from eigenvectors
       (floating point rounding errors from the solver)
    2. Resampling to 1-hour before Johansen estimation
       removes microstructure noise from 1-minute data
    """

    SIGNIFICANCE_TO_CV_INDEX = {
        0.10: 0,
        0.05: 1,
        0.01: 2
    }

    def __init__(self, config: dict):
        self.config    = config
        self.coint_cfg = config["cointegration"]

        self.significance = self.coint_cfg["significance_level"]
        self.max_lags     = self.coint_cfg["max_lags"]
        self.window_days  = self.coint_cfg["window_days"]
        self.min_obs      = self.coint_cfg["min_observations"]
        self.cases        = self.coint_cfg.get("cases", {})

        self.cv_index = self.SIGNIFICANCE_TO_CV_INDEX.get(
            self.significance, 0
        )

        logger.info(
            f"JohansenCointegration initialized | "
            f"significance={self.significance} | "
            f"cv_index={self.cv_index} | "
            f"cases={list(self.cases.keys())}"
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _asset_name_to_key(
        self,
        asset: str,
        exchange: str = "binance"
    ) -> str:
        """
        Convert config asset name to ohlcv_dict key.
        "BTC-USDT" -> "binance_BTCUSDT"
        """
        return f"{exchange}_{asset.replace('-', '')}"

    def _extract_price_matrix(
        self,
        ohlcv_dict: dict,
        asset_keys: list
    ) -> Optional[pd.DataFrame]:
        """
        Build log price matrix for Johansen estimation.

        Resamples to 1-hour frequency before running the test.

        Why 1-hour and not 1-minute:
        Johansen tests for cointegration in price LEVELS.
        At 1-minute frequency, microstructure noise
        (bid-ask bounce, latency differences between exchanges)
        creates spurious autocorrelation in residuals,
        violating the test's assumptions.

        Hourly prices smooth out this noise while retaining
        the structural relationship between assets.
        The cointegrating vector estimated on hourly data
        is then applied to minute-level prices for
        spread computation and signal generation.
        """
        price_data = {}

        for key in asset_keys:
            if key not in ohlcv_dict:
                logger.error(f"Asset key not found: {key}")
                return None

            df = ohlcv_dict[key]

            if "close" not in df.columns:
                logger.error(f"No close column for {key}")
                return None

            close = df["close"].replace(0, np.nan).dropna()

            # Resample to 1-hour — take last price in each hour
            if hasattr(close.index, 'freq') or isinstance(
                close.index, pd.DatetimeIndex
            ):
                close_hourly = close.resample("1h").last().dropna()
            else:
                close_hourly = close

            price_data[key] = np.log(close_hourly)

        price_matrix = pd.DataFrame(price_data).dropna()

        if len(price_matrix) < self.min_obs:
            logger.warning(
                f"Insufficient observations after resampling: "
                f"{len(price_matrix)} < {self.min_obs}"
            )
            return None

        logger.debug(
            f"Price matrix: {price_matrix.shape} "
            f"(1h frequency, log prices)"
        )

        return price_matrix

    def _run_johansen(
        self,
        price_matrix: pd.DataFrame
    ) -> Optional[object]:
        """Run Johansen test on price matrix."""
        try:
            result = coint_johansen(
                price_matrix,
                det_order=0,
                k_ar_diff=self.max_lags
            )
            return result
        except Exception as e:
            logger.error(f"Johansen test failed: {e}")
            return None

    def _count_cointegrating_vectors(
        self,
        result: object
    ) -> int:
        """
        Count significant cointegrating vectors via trace test.

        Uses np.real() to handle any complex residuals
        from the eigenvalue computation.
        """
        # Force real — imaginary parts are numerical noise
        trace_stats = np.real(result.lr1)
        crit_vals   = np.real(result.cvt)
        n_vectors   = 0

        for r in range(len(trace_stats)):
            stat = trace_stats[r]
            cv   = crit_vals[r, self.cv_index]

            if stat > cv:
                n_vectors += 1
                logger.debug(
                    f"r={r}: stat={stat:.4f} > cv={cv:.4f} "
                    f"-> reject H0"
                )
            else:
                logger.debug(
                    f"r={r}: stat={stat:.4f} <= cv={cv:.4f} "
                    f"-> fail to reject H0"
                )
                break

        logger.info(
            f"Cointegrating vectors found: {n_vectors}"
        )
        return n_vectors

    def _extract_cointegrating_vector(
        self,
        result:     object,
        n_vectors:  int,
        asset_keys: list
    ) -> Optional[pd.Series]:
        """
        Extract and normalise the first cointegrating vector.

        CRITICAL FIX: np.real() strips imaginary parts.

        Johansen eigenvectors are mathematically real for
        real-valued price data. The imaginary parts that
        appear (e.g. 1.0000+0.0000j) are floating point
        rounding errors from the numerical eigenvalue solver.

        Without np.real():
        - The vector has dtype complex128
        - Dot product with prices produces complex spread
        - pandas rolling() fails on complex series
        - All z-scores become NaN
        - Zero trades

        With np.real():
        - Vector is dtype float64
        - Everything downstream works correctly
        """
        if n_vectors == 0:
            return None

        evec       = result.evec
        raw_vector = evec[:, 0]

        # THE FIX: strip imaginary parts (they are rounding errors)
        raw_vector = np.real(raw_vector)

        # Normalize: first asset coefficient = 1.0
        if abs(raw_vector[0]) < 1e-10:
            logger.warning(
                "First eigenvector component near zero — "
                "normalization unstable"
            )
            return None

        normalized   = raw_vector / raw_vector[0]
        coint_vector = pd.Series(normalized, index=asset_keys)

        logger.info(
            f"Cointegrating vector: "
            + " | ".join(
                f"{k.split('_')[-1]}={v:.4f}"
                for k, v in coint_vector.items()
            )
        )

        return coint_vector

    # ------------------------------------------------------------------ #
    #  Public interface — single case                                     #
    # ------------------------------------------------------------------ #

    def test(
        self,
        ohlcv_dict: dict,
        asset_keys: list
    ) -> Tuple[bool, Optional[pd.Series], int]:
        """
        Run Johansen test for one specific case.
        Returns (is_cointegrated, coint_vector, n_vectors).
        """
        price_matrix = self._extract_price_matrix(
            ohlcv_dict, asset_keys
        )
        if price_matrix is None:
            return False, None, 0

        result = self._run_johansen(price_matrix)
        if result is None:
            return False, None, 0

        n_vectors    = self._count_cointegrating_vectors(result)
        coint_vector = self._extract_cointegrating_vector(
            result, n_vectors, asset_keys
        )

        return n_vectors > 0, coint_vector, n_vectors

    # ------------------------------------------------------------------ #
    #  Public interface — all cases                                       #
    # ------------------------------------------------------------------ #

    def test_all_cases(
        self,
        ohlcv_dict:       dict,
        primary_exchange: str = "binance"
    ) -> Dict[str, dict]:
        """
        Test all configured cases simultaneously.

        Returns dict of only the cointegrated cases:
        {
            "BTC-ETH": {
                "coint_vector": pd.Series,
                "asset_keys":   list,
                "n_vectors":    int,
                "assets":       list,
            },
            ...
        }
        """
        cointegrated_cases = {}

        for case_name, assets in self.cases.items():

            asset_keys = [
                self._asset_name_to_key(a, primary_exchange)
                for a in assets
            ]

            logger.info(
                f"Testing case: {case_name} | {asset_keys}"
            )

            # Check all keys exist
            missing = [k for k in asset_keys if k not in ohlcv_dict]
            if missing:
                logger.warning(
                    f"Case {case_name}: missing keys {missing} — skipping"
                )
                continue

            is_coint, coint_vector, n_vectors = self.test(
                ohlcv_dict, asset_keys
            )

            if is_coint and coint_vector is not None:
                cointegrated_cases[case_name] = {
                    "coint_vector": coint_vector,
                    "asset_keys":   asset_keys,
                    "n_vectors":    n_vectors,
                    "assets":       assets,
                }
                logger.info(
                    f"Case {case_name}: COINTEGRATED "
                    f"({n_vectors} vectors)"
                )
            else:
                logger.info(
                    f"Case {case_name}: not cointegrated "
                )

        logger.info(
            f"test_all_cases complete | "
            f"{len(cointegrated_cases)}/{len(self.cases)} "
            f"cases cointegrated"
        )

        return cointegrated_cases

    # ------------------------------------------------------------------ #
    #  Public interface — rolling test                                    #
    # ------------------------------------------------------------------ #

    def rolling_test(
        self,
        ohlcv_dict:       dict,
        asset_keys:       list,
        interval_minutes: int = 1
    ) -> pd.DataFrame:
        """
        Rolling Johansen on hourly data, re-estimated daily.

        window_size = 30 days * 24 hours = 720 hourly rows
        step_size   = 1 day  * 24 hours  = 24 hourly rows

        Note: interval_minutes kept for interface compatibility
        but internally always uses hourly data.
        """
        hours_per_day = 24
        window_size   = self.window_days * hours_per_day   # 720
        step_size     = hours_per_day                       # 24

        price_matrix = self._extract_price_matrix(
            ohlcv_dict, asset_keys
        )
        if price_matrix is None:
            return pd.DataFrame()

        total_rows = len(price_matrix)
        records    = []

        for end_idx in range(window_size, total_rows, step_size):

            start_idx  = end_idx - window_size
            window_df  = price_matrix.iloc[start_idx:end_idx]
            window_end = price_matrix.index[end_idx - 1]

            result = self._run_johansen(window_df)

            if result is None:
                records.append({
                    "timestamp":       window_end,
                    "is_cointegrated": False,
                    "n_vectors":       0,
                    **{k: np.nan for k in asset_keys}
                })
                continue

            n_vectors    = self._count_cointegrating_vectors(result)
            coint_vector = self._extract_cointegrating_vector(
                result, n_vectors, asset_keys
            )

            if coint_vector is not None:
                record = {
                    "timestamp":       window_end,
                    "is_cointegrated": True,
                    "n_vectors":       n_vectors,
                    **coint_vector.to_dict()
                }
            else:
                record = {
                    "timestamp":       window_end,
                    "is_cointegrated": False,
                    "n_vectors":       0,
                    **{k: np.nan for k in asset_keys}
                }

            records.append(record)

        if not records:
            return pd.DataFrame()

        results_df = pd.DataFrame(records).set_index("timestamp")

        coint_rate = results_df["is_cointegrated"].mean()
        logger.info(
            f"Rolling test | {asset_keys} | "
            f"{len(results_df)} daily windows | "
            f"cointegration_rate={coint_rate:.1%}"
        )

        return results_df

    def rolling_test_all_cases(
        self,
        ohlcv_dict:       dict,
        primary_exchange: str = "binance",
        interval_minutes: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Run rolling Johansen test for ALL cases.
        Returns dict keyed by case name.
        """
        all_rolling = {}

        for case_name, assets in self.cases.items():

            asset_keys = [
                self._asset_name_to_key(a, primary_exchange)
                for a in assets
            ]

            missing = [k for k in asset_keys if k not in ohlcv_dict]
            if missing:
                logger.warning(
                    f"Rolling test {case_name}: "
                    f"missing keys {missing} — skipping"
                )
                continue

            logger.info(
                f"Rolling test: {case_name} | "
                f"assets={asset_keys}"
            )

            rolling_df = self.rolling_test(
                ohlcv_dict,
                asset_keys,
                interval_minutes
            )

            if not rolling_df.empty:
                all_rolling[case_name] = rolling_df

        logger.info(
            f"Rolling test complete for "
            f"{len(all_rolling)} cases"
        )

        return all_rolling

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class SpreadBuilder:
    """
    Responsible for one thing only:
    given a cointegrating vector and aligned price data,
    construct the spread time series and compute its
    z-score at every timestamp.

    Two build modes:

    1. build()
       Uses a rolling window z-score.
       Used for training data analysis only.
       Requires enough rows to fill the rolling window.

    2. build_with_baseline()
       Uses fixed mean/std from training data.
       Used for TEST windows in walk-forward backtest.
       Works correctly even on 7-day test windows.
       This is the correct walk-forward approach.

    Key fix: np.real() applied to all spread values
    to handle complex residuals from Johansen eigenvectors.
    """

    def __init__(self, config: dict):
        self.config    = config
        self.coint_cfg = config["cointegration"]
        self.mkt_cfg   = config["market_data"]

        self.window_days = self.coint_cfg["window_days"]

        logger.info("SpreadBuilder initialized")

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_log_prices(
        self,
        ohlcv_dict: dict,
        asset_keys: list
    ) -> Optional[pd.DataFrame]:
        """
        Extract log close prices for all assets into one DataFrame.

        Uses 1-minute data for spread construction — the full
        granularity for signal generation and execution.

        Note: Johansen estimation uses 1-hour resampled data,
        but the spread itself is computed at 1-minute resolution
        so that entry/exit signals fire at the correct moments.
        """
        log_prices = {}

        for key in asset_keys:
            if key not in ohlcv_dict:
                logger.error(f"Key not found in OHLCV dict: {key}")
                return None

            df = ohlcv_dict[key]

            if "close" not in df.columns:
                logger.error(f"No close column for {key}")
                return None

            close = df["close"].replace(0, np.nan).dropna()
            log_prices[key] = np.log(close)

        price_matrix = pd.DataFrame(log_prices).dropna()

        logger.debug(
            f"Log price matrix: {price_matrix.shape} | "
            f"assets={asset_keys}"
        )

        return price_matrix

    def _compute_raw_spread(
        self,
        price_matrix: pd.DataFrame,
        coint_vector: pd.Series
    ) -> pd.Series:
        """
        Compute spread as dot product of log prices and vector.

        spread = 1.0*log(BTC) + (-0.6)*log(ETH) + (-0.3)*log(SOL)

        CRITICAL FIX: np.real() applied to output.
        If the cointegrating vector contains any complex residuals
        (floating point noise from Johansen eigensolver), the dot
        product produces a complex spread. np.real() strips the
        imaginary part safely — it is always effectively zero.
        """
        aligned_vector = coint_vector.reindex(price_matrix.columns)

        if aligned_vector.isna().any():
            missing = aligned_vector[aligned_vector.isna()].index.tolist()
            logger.error(
                f"Cointegrating vector missing keys: {missing}"
            )
            return pd.Series(dtype=float)

        # Matrix multiply: (T x N) @ (N,) -> (T,)
        spread = price_matrix @ aligned_vector.values

        # CRITICAL: strip any complex residuals
        spread = pd.Series(
            np.real(spread.values),
            index=price_matrix.index
        )

        logger.debug(
            f"Raw spread computed | "
            f"mean={spread.mean():.6f} | "
            f"std={spread.std():.6f} | "
            f"rows={len(spread)}"
        )

        return spread

    def _compute_zscore_rolling(
        self,
        spread: pd.Series,
        window_rows: int
    ) -> pd.Series:
        """
        Compute rolling z-score of the spread.

        Used for training data analysis only.
        Requires enough rows to fill the rolling window.
        """
        rolling_mean = spread.rolling(
            window=window_rows,
            min_periods=window_rows // 2
        ).mean()

        rolling_std = spread.rolling(
            window=window_rows,
            min_periods=window_rows // 2
        ).std()

        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (spread - rolling_mean) / rolling_std

        nan_count = zscore.isna().sum()
        logger.debug(
            f"Z-score computed | "
            f"NaN warmup rows={nan_count} | "
            f"valid rows={len(zscore) - nan_count}"
        )

        return zscore

    def _compute_zscore_fixed(
        self,
        spread:     pd.Series,
        train_mean: float,
        train_std:  float
    ) -> pd.Series:
        """
        Compute z-score using fixed training statistics.

        Used for test windows in walk-forward backtest.

        Why this is correct:
        In a walk-forward system, the model is calibrated on
        training data. The z-score measures how many standard
        deviations the current spread is from the training mean.
        Using test-window rolling statistics would look forward
        and constitute look-ahead bias.

        Using fixed training statistics:
        - No warmup period required
        - Every test row produces a valid z-score
        - Strictly uses only pre-test information
        """
        const_min_std = 1e-10

        if train_std < const_min_std or np.isnan(train_std):
            logger.warning(
                f"Invalid train_std={train_std:.6f} — "
                f"cannot compute z-score"
            )
            return pd.Series(np.nan, index=spread.index)

        zscore = (spread - train_mean) / train_std

        valid = zscore.notna().sum()
        logger.debug(
            f"Z-score (fixed baseline) | "
            f"valid rows={valid} | "
            f"mean_z={zscore.mean():.4f} | "
            f"std_z={zscore.std():.4f}"
        )

        return zscore

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def build(
        self,
        ohlcv_dict:   dict,
        asset_keys:   list,
        coint_vector: pd.Series
    ) -> pd.DataFrame:
        """
        Build spread and z-score using rolling window statistics.

        Use this for:
        - Training data analysis
        - Quick exploration
        - Benchmarking against static-vector baseline

        NOT suitable for short test windows (7 days) because
        the rolling window requires 30 days of data to warm up.
        Use build_with_baseline() for test windows instead.

        Returns
        -------
        pd.DataFrame with columns: spread, zscore
        """
        logger.info(
            f"Building spread (rolling z-score) | "
            f"assets={asset_keys}"
        )

        price_matrix = self._get_log_prices(ohlcv_dict, asset_keys)
        if price_matrix is None:
            return pd.DataFrame()

        spread = self._compute_raw_spread(price_matrix, coint_vector)
        if spread.empty:
            return pd.DataFrame()

        window_rows = self.window_days * 1440
        zscore      = self._compute_zscore_rolling(spread, window_rows)

        result = pd.DataFrame({
            "spread": spread,
            "zscore": zscore,
        })

        logger.info(
            f"Spread built | rows={len(result)} | "
            f"spread_mean={spread.mean():.6f} | "
            f"zscore_std={zscore.std():.4f}"
        )

        return result

    def build_with_baseline(
        self,
        ohlcv_dict:   dict,
        asset_keys:   list,
        coint_vector: pd.Series,
        train_mean:   float,
        train_std:    float
    ) -> pd.DataFrame:
        """
        Build spread for TEST window using training statistics.

        This is the correct method for walk-forward backtesting.

        The z-score is computed as:
            z = (spread - train_mean) / train_std

        Where train_mean and train_std come from the training
        window that preceded this test window.

        Benefits over rolling z-score on test data:
        1. No warmup period — every row produces a valid z-score
        2. No look-ahead bias — uses only pre-test statistics
        3. Consistent with how the model was calibrated

        Parameters
        ----------
        ohlcv_dict   : dict of DataFrames (test period prices)
        asset_keys   : list of keys matching ohlcv_dict
        coint_vector : hedge ratios from training Johansen
        train_mean   : mean of spread during training period
        train_std    : std of spread during training period

        Returns
        -------
        pd.DataFrame with columns: spread, zscore
        All rows have valid z-scores (no NaN warmup).
        """
        logger.info(
            f"Building spread (fixed baseline) | "
            f"assets={asset_keys} | "
            f"train_mean={train_mean:.6f} | "
            f"train_std={train_std:.6f}"
        )

        price_matrix = self._get_log_prices(ohlcv_dict, asset_keys)
        if price_matrix is None:
            return pd.DataFrame()

        spread = self._compute_raw_spread(price_matrix, coint_vector)
        if spread.empty:
            return pd.DataFrame()

        zscore = self._compute_zscore_fixed(
            spread, train_mean, train_std
        )

        result = pd.DataFrame({
            "spread": spread,
            "zscore": zscore,
        })

        valid_zscores = zscore.notna().sum()
        logger.info(
            f"Spread built (fixed baseline) | "
            f"rows={len(result)} | "
            f"valid_zscores={valid_zscores} | "
            f"spread_mean={spread.mean():.6f}"
        )

        return result

    def build_rolling(
        self,
        ohlcv_dict:     dict,
        asset_keys:     list,
        rolling_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build spread using time-varying cointegrating vectors
        from rolling Johansen estimation.

        Used for training period analysis and the static
        OU benchmark strategy.
        """
        logger.info(
            f"Building rolling spread | assets={asset_keys} | "
            f"windows={len(rolling_results)}"
        )

        price_matrix = self._get_log_prices(ohlcv_dict, asset_keys)
        if price_matrix is None:
            return pd.DataFrame()

        spread_values = pd.Series(np.nan, index=price_matrix.index)

        vector_cols = [
            c for c in asset_keys
            if c in rolling_results.columns
        ]

        if len(vector_cols) != len(asset_keys):
            missing = set(asset_keys) - set(vector_cols)
            logger.error(f"Missing vector columns: {missing}")
            return pd.DataFrame()

        window_dates = rolling_results.index

        for i, window_end in enumerate(window_dates):

            vector_row = rolling_results.loc[window_end, vector_cols]

            if vector_row.isna().any():
                continue

            apply_start = window_end
            apply_end   = (
                window_dates[i + 1]
                if i + 1 < len(window_dates)
                else price_matrix.index[-1]
            )

            mask = (
                (price_matrix.index >= apply_start) &
                (price_matrix.index <  apply_end)
            )
            window_prices = price_matrix[mask]

            if window_prices.empty:
                continue

            coint_vector = pd.Series(
                np.real(vector_row.values),
                index=vector_cols
            )
            window_spread = window_prices @ coint_vector.values
            spread_values[window_prices.index] = np.real(
                window_spread
            )

        filled = spread_values.notna().sum()
        logger.info(
            f"Rolling spread applied | "
            f"{filled} / {len(spread_values)} rows filled"
        )

        window_rows = self.window_days * 1440
        zscore      = self._compute_zscore_rolling(
            spread_values, window_rows
        )

        is_coint_daily = rolling_results["is_cointegrated"]
        is_coint_minute = is_coint_daily.reindex(
            price_matrix.index, method="ffill"
        )

        result = pd.DataFrame({
            "spread":          spread_values,
            "zscore":          zscore,
            "is_cointegrated": is_coint_minute,
        })

        return result

    def get_hedge_ratios(
        self,
        coint_vector: pd.Series,
        asset_keys:   list
    ) -> dict:
        """Return hedge ratios in human-readable format."""
        ratios = {}
        for key in asset_keys:
            if key in coint_vector.index:
                ratios[key] = round(float(coint_vector[key]), 6)

        logger.info(
            "Hedge ratios: " +
            " | ".join(f"{k}={v}" for k, v in ratios.items())
        )
        return ratios

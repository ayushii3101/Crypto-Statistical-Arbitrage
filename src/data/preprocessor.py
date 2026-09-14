import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self, config: dict):
        self.config   = config
        self.path_cfg = config["paths"]
        self.mkt_cfg  = config["market_data"]

        logger.info("DataPreprocessor initialized")

    def _fill_missing_candles(
        self,
        df: pd.DataFrame,
        interval_minutes: int = 1
    ) -> pd.DataFrame:

        if df.empty:
            return df

        freq = f"{interval_minutes}min"
        complete_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
            tz=df.index.tz       
        )

        missing_count = len(complete_index) - len(df)

        if missing_count > 0:
            logger.warning(
                f"Filling {missing_count} missing candles "
                f"({missing_count / len(complete_index):.3%} of total)"
            )

        df = df.reindex(complete_index)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        if "close" in df.columns:
            for col in ["open", "high", "low"]:
                if col in df.columns:
                    still_nan_volume = df["volume"].isna()
                    df.loc[still_nan_volume, col] = df.loc[
                        still_nan_volume, "close"
                    ]

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)

        return df

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        column: str = "close",
        window: int = 60,
        z_threshold: float = 10.0
    ) -> pd.DataFrame:

        if column not in df.columns:
            return df

        rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
        rolling_std  = df[column].rolling(window=window, min_periods=1).std()

        # Avoid division by zero when std is 0 (flat price periods)
        rolling_std = rolling_std.replace(0, np.nan)

        z_scores = (df[column] - rolling_mean) / rolling_std

        outlier_mask = z_scores.abs() > z_threshold
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.warning(
                f"Replacing {outlier_count} outliers in '{column}' "
                f"(|z| > {z_threshold})"
            )
            df[column] = df[column].where(~outlier_mask)
            df[column] = df[column].ffill()

        return df

    def _align_timestamps(
        self,
        dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        
        if not dfs:
            return dfs

        # Find the common timestamp index across all DataFrames
        common_index = None
        for key, df in dfs.items():
            if common_index is None:
                common_index = df.index
            else:
                # Intersection keeps only timestamps in both
                common_index = common_index.intersection(df.index)

        logger.info(
            f"Timestamp alignment: {len(common_index)} common rows "
            f"across {len(dfs)} DataFrames"
        )

        aligned = {}
        for key, df in dfs.items():
            original_len = len(df)
            aligned[key] = df.reindex(common_index)
            dropped = original_len - len(common_index)
            if dropped > 0:
                logger.warning(
                    f"{key}: dropped {dropped} rows during alignment"
                )
            aligned[key] = df.reindex(common_index)

        return aligned

    def _compute_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close"
    ) -> pd.DataFrame:
       
        if price_col not in df.columns:
            logger.warning(f"Column '{price_col}' not found for return computation")
            return df

        # log(P_t) - log(P_{t-1}) = log(P_t / P_{t-1})
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

        # First row is always NaN — drop it cleanly
        df = df.dropna(subset=["log_return"])

        logger.debug(
            f"Log returns computed: {len(df)} rows, "
            f"mean={df['log_return'].mean():.6f}, "
            f"std={df['log_return'].std():.6f}"
        )

        return df

    def _normalize_symbol_format(self, asset: str) -> str:
     
        return asset.replace("-", "")

    def preprocess_ohlcv(
        self,
        df: pd.DataFrame,
        label: str = "OHLCV"
    ) -> pd.DataFrame:
    
        logger.info(f"Preprocessing OHLCV: {label}")
        original_len = len(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = self._fill_missing_candles(df, interval_minutes=1)

        df = self._remove_outliers(df, column="close")

        df = self._compute_returns(df, price_col="close")

        logger.info(
            f"Preprocessing complete: {label} | "
            f"{original_len} → {len(df)} rows"
        )

        return df

    def preprocess_funding_rates(
        self,
        df: pd.DataFrame,
        label: str = "FundingRate"
    ) -> pd.DataFrame:
        """
        Preprocessing pipeline for funding rate data.

        Funding rates need less processing than OHLCV:
        - Fill small gaps with forward-fill
          (funding rate persists until next update)
        - No outlier removal beyond what validator flagged
          (extreme funding rates are real signals, not errors)
        - No return computation (rates are already relative values)
        """
        logger.info(f"Preprocessing funding rates: {label}")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, format='ISO8601', utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Funding rates: fill gaps by carrying last known rate forward
        # This is correct — the rate stays fixed until the next
        # 8-hour settlement
        df["funding_rate"] = df["funding_rate"].ffill()

        # Compute cumulative funding cost — useful for cost modeling later
        # This is the running sum of all funding paid/received
        df["cumulative_funding"] = df["funding_rate"].cumsum()

        logger.info(f"Funding rate preprocessing complete: {label}")

        return df

    def align_all(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all OHLCV DataFrames to a common timestamp index.

        Call this after preprocessing each DataFrame individually.
        Alignment must happen AFTER gap-filling — otherwise the
        gaps would cause unnecessary row drops during intersection.

        Parameters
        ----------
        ohlcv_dict : Dict[str, pd.DataFrame]
            Keys are descriptive labels like "binance_BTC"
            Values are preprocessed OHLCV DataFrames

        Returns
        -------
        Dict[str, pd.DataFrame]
            Same keys, all DataFrames on identical timestamp index
        """
        logger.info(
            f"Aligning {len(ohlcv_dict)} DataFrames to common index..."
        )
        return self._align_timestamps(ohlcv_dict)

    def preprocess_all(self) -> Tuple[Dict, Dict]:
        """
        Load all raw files, preprocess each one, align timestamps,
        and save to data/processed/.

        Returns
        -------
        ohlcv_dict    : Dict[str, pd.DataFrame]
        funding_dict  : Dict[str, pd.DataFrame]

        Both dicts are keyed by "{exchange}_{asset}" for easy lookup.
        Example key: "binance_BTCUSDT"
        """
        raw_dir       = self.path_cfg["raw_data_dir"]
        processed_dir = self.path_cfg["processed_data_dir"]
        assets        = self.mkt_cfg["assets"]
        exchanges     = self.mkt_cfg["exchanges"]
        interval      = self.mkt_cfg["interval"]

        os.makedirs(processed_dir, exist_ok=True)

        ohlcv_dict   = {}
        funding_dict = {}

        # ── Step 1: preprocess each file individually ──────────────────
        for exchange_name in exchanges:
            for asset in assets:

                safe_asset = self._normalize_symbol_format(asset)
                key        = f"{exchange_name}_{safe_asset}"

                # Load and preprocess OHLCV
                ohlcv_file = os.path.join(
                    raw_dir,
                    f"{exchange_name}_{safe_asset}_{interval}_ohlcv.csv"
                )

                if os.path.exists(ohlcv_file):
                    df_raw = pd.read_csv(
                        ohlcv_file,
                        index_col=0,
                        parse_dates=True
                    )
                    label         = f"{exchange_name}|{asset}"
                    df_clean      = self.preprocess_ohlcv(df_raw, label)
                    ohlcv_dict[key] = df_clean
                    logger.info(f"OHLCV preprocessed: {key}")
                else:
                    logger.error(f"OHLCV file not found: {ohlcv_file}")

                # Load and preprocess funding rates
                funding_file = os.path.join(
                    raw_dir,
                    f"{exchange_name}_{safe_asset}_funding_rate.csv"
                )

                if os.path.exists(funding_file):
                    df_raw = pd.read_csv(
                        funding_file,
                        index_col=0,
                        parse_dates=True
                    )
                    label             = f"{exchange_name}|{asset}|funding"
                    df_clean          = self.preprocess_funding_rates(
                        df_raw, label
                    )
                    funding_dict[key] = df_clean
                    logger.info(f"Funding rates preprocessed: {key}")
                else:
                    logger.error(f"Funding file not found: {funding_file}")

        # ── Step 2: align all OHLCV to common timestamp index ──────────
        # Must happen AFTER individual preprocessing
        # so gaps are filled before intersection is computed
        logger.info("Aligning all OHLCV DataFrames...")
        ohlcv_dict = self.align_all(ohlcv_dict)

        # ── Step 3: save processed files ────────────────────────────────
        for key, df in ohlcv_dict.items():
            out_path = os.path.join(
                processed_dir,
                f"{key}_{interval}_processed.csv"
            )
            df.to_csv(out_path)
            logger.info(f"Saved processed OHLCV → {out_path}")

        for key, df in funding_dict.items():
            out_path = os.path.join(
                processed_dir,
                f"{key}_funding_processed.csv"
            )
            df.to_csv(out_path)
            logger.info(f"Saved processed funding → {out_path}")

        logger.info(
            f"preprocess_all complete | "
            f"{len(ohlcv_dict)} OHLCV + {len(funding_dict)} funding files"
        )

        return ohlcv_dict, funding_dict

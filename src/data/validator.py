
import os
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataValidator:
    MIN_PRICE    = 0.0
    MAX_PRICE    = 1_000_000.0   # $1M per BTC is our sanity ceiling
    MIN_ROWS     = 100
    MAX_GAP_MINUTES = 5
    MAX_NAN_FRACTION = 0.01      # 1%

    def __init__(self, config: dict):
        self.config   = config
        self.path_cfg = config["paths"]
        self.mkt_cfg  = config["market_data"]

        logger.info("DataValidator initialized")

    def _check_not_empty(
        self,
        df: pd.DataFrame,
        label: str
    ) -> Tuple[bool, str]:
 
        if df.empty:
            return False, f"{label}: DataFrame is completely empty"

        if len(df) < self.MIN_ROWS:
            return (
                False,
                f"{label}: only {len(df)} rows, "
                f"minimum required is {self.MIN_ROWS}"
            )

        return True, f"{label}: row count OK ({len(df)} rows)"
    
    def _check_no_nulls(
        self,
        df: pd.DataFrame,
        label: str
    ) -> Tuple[bool, str]:
        
        null_fractions = df.isnull().mean()   
        bad_columns    = null_fractions[
            null_fractions > self.MAX_NAN_FRACTION
        ]

        if not bad_columns.empty:
            details = ", ".join(
                f"{col}={frac:.2%}"
                for col, frac in bad_columns.items()
            )
            return (
                False,
                f"{label}: NaN fraction too high → {details}"
            )

        return True, f"{label}: NaN check OK"
    
    def _check_no_infinite(
        self,
        df: pd.DataFrame,
        label: str
    ) -> Tuple[bool, str]:

        numeric_df = df.select_dtypes(include=[np.number])
        inf_mask   = np.isinf(numeric_df)

        if inf_mask.any().any():
            bad_cols = inf_mask.any()
            bad_cols = bad_cols[bad_cols].index.tolist()
            return (
                False,
                f"{label}: infinite values found in columns: {bad_cols}"
            )

        return True, f"{label}: no infinite values"
    
    def _check_price_domain(
        self,
        df: pd.DataFrame,
        label: str
    ) -> Tuple[bool, str]:
       
        price_cols = ["open", "high", "low", "close"]
        errors     = []

        for col in price_cols:
            if col not in df.columns:
                continue
            if (df[col] <= self.MIN_PRICE).any():
                count = (df[col] <= self.MIN_PRICE).sum()
                errors.append(
                    f"{col} has {count} values <= 0"
                )

        for col in price_cols:
            if col not in df.columns:
                continue
            if (df[col] > self.MAX_PRICE).any():
                count = (df[col] > self.MAX_PRICE).sum()
                errors.append(
                    f"{col} has {count} values > {self.MAX_PRICE}"
                )

        if all(c in df.columns for c in price_cols):
            bad_hl = (df["high"] < df["low"]).sum()
            if bad_hl > 0:
                errors.append(
                    f"high < low in {bad_hl} candles"
                )

            bad_h = (
                (df["high"] < df["open"]) |
                (df["high"] < df["close"])
            ).sum()
            if bad_h > 0:
                errors.append(
                    f"high < open or close in {bad_h} candles"
                )

            bad_l = (
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            ).sum()
            if bad_l > 0:
                errors.append(
                    f"low > open or close in {bad_l} candles"
                )

        if "volume" in df.columns:
            if (df["volume"] < 0).any():
                count = (df["volume"] < 0).sum()
                errors.append(f"volume has {count} negative values")

        if errors:
            return False, f"{label}: domain violations → " + "; ".join(errors)

        return True, f"{label}: price domain OK"
    
    def _check_timestamp_gaps(
        self,
        df: pd.DataFrame,
        label: str,
        interval_minutes: int = 1
    ) -> Tuple[bool, str]:
    
        if len(df) < 2:
            return True, f"{label}: not enough rows to check gaps"

        time_diffs = df.index.to_series().diff().dropna()
        time_diffs_min = time_diffs.dt.total_seconds() / 60

        large_gaps = time_diffs_min[
            time_diffs_min > self.MAX_GAP_MINUTES
        ]

        if not large_gaps.empty:
            max_gap   = large_gaps.max()
            gap_count = len(large_gaps)
            worst_gap_ts = large_gaps.idxmax()
            return (
                False,
                f"{label}: {gap_count} timestamp gap(s) found, "
                f"largest = {max_gap:.1f} min at {worst_gap_ts}"
            )

        return True, f"{label}: timestamp continuity OK"
    
    def _check_funding_rate_domain(
        self,
        df: pd.DataFrame,
        label: str
    ) -> Tuple[bool, str]:
     
        if "funding_rate" not in df.columns:
            return False, f"{label}: funding_rate column missing"

        MAX_FUNDING = 0.0075    
        MIN_FUNDING = -0.0075

        out_of_bounds = (
            (df["funding_rate"] > MAX_FUNDING) |
            (df["funding_rate"] < MIN_FUNDING)
        )

        if out_of_bounds.any():
            count = out_of_bounds.sum()
            extreme_val = df["funding_rate"][out_of_bounds].abs().max()
            return (
                False,
                f"{label}: {count} funding rates outside "
                f"[{MIN_FUNDING}, {MAX_FUNDING}], "
                f"max absolute value = {extreme_val:.6f}"
            )
        return True, f"{label}: funding rate domain OK"
    
    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        label: str = "OHLCV"
    ) -> Dict:
   
        checks = [
            ("not_empty",        self._check_not_empty(df, label)),
            ("no_nulls",         self._check_no_nulls(df, label)),
            ("no_infinite",      self._check_no_infinite(df, label)),
            ("price_domain",     self._check_price_domain(df, label)),
            ("timestamp_gaps",   self._check_timestamp_gaps(df, label)),
        ]

        results  = []
        all_pass = True

        for check_name, (passed, message) in checks:
            if not passed:
                all_pass = False
                logger.warning(f"FAIL [{check_name}] {message}")
            else:
                logger.debug(f"PASS [{check_name}] {message}")

            results.append({
                "name":    check_name,
                "passed":  passed,
                "message": message
            })

        status = "PASSED" if all_pass else "FAILED"
        logger.info(f"Validation {status} for {label}")

        return {
            "passed": all_pass,
            "label":  label,
            "checks": results
        }
    
    def validate_funding_rates(
        self,
        df: pd.DataFrame,
        label: str = "FundingRate"
    ) -> Dict:
       
        checks = [
            ("not_empty",           self._check_not_empty(df, label)),
            ("no_nulls",            self._check_no_nulls(df, label)),
            ("no_infinite",         self._check_no_infinite(df, label)),
            ("funding_rate_domain", self._check_funding_rate_domain(df, label)),
            ("timestamp_gaps",      self._check_timestamp_gaps(
                                        df, label,
                                        interval_minutes=480  # 8 hours
                                    )),
        ]

        results  = []
        all_pass = True

        for check_name, (passed, message) in checks:
            if not passed:
                all_pass = False
                logger.warning(f"FAIL [{check_name}] {message}")
            else:
                logger.debug(f"PASS [{check_name}] {message}")

            results.append({
                "name":    check_name,
                "passed":  passed,
                "message": message
            })

        status = "PASSED" if all_pass else "FAILED"
        logger.info(f"Validation {status} for {label}")

        return {
            "passed": all_pass,
            "label":  label,
            "checks": results
        }
    
    def validate_all(self) -> Dict[str, Dict]:
      
        raw_dir  = self.path_cfg["raw_data_dir"]
        assets   = self.mkt_cfg["assets"]
        exchanges = self.mkt_cfg["exchanges"]
        interval = self.mkt_cfg["interval"]

        all_reports = {}
        total       = 0
        passed      = 0

        logger.info(f"Starting validate_all from {raw_dir}")

        for exchange_name in exchanges:
            for asset in assets:

                safe_asset = asset.replace("-", "")

                ohlcv_file = (
                    f"{exchange_name}_{safe_asset}"
                    f"_{interval}_ohlcv.csv"
                )
                ohlcv_path = os.path.join(raw_dir, ohlcv_file)

                total += 1
                if not os.path.exists(ohlcv_path):
                    # File missing entirely — critical failure
                    report = {
                        "passed": False,
                        "label":  ohlcv_file,
                        "checks": [{
                            "name":    "file_exists",
                            "passed":  False,
                            "message": f"File not found: {ohlcv_path}"
                        }]
                    }
                    logger.error(f"File missing: {ohlcv_path}")
                else:
                    df = pd.read_csv(ohlcv_path, index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True)
                    label = f"{exchange_name}|{asset}|OHLCV"
                    report = self.validate_ohlcv(df, label)
                    if report["passed"]:
                        passed += 1

                all_reports[ohlcv_file] = report

                funding_file = (
                    f"{exchange_name}_{safe_asset}"
                    f"_funding_rate.csv"
                )
                funding_path = os.path.join(raw_dir, funding_file)

                total += 1
                if not os.path.exists(funding_path):
                    report = {
                        "passed": False,
                        "label":  funding_file,
                        "checks": [{
                            "name":    "file_exists",
                            "passed":  False,
                            "message": f"File not found: {funding_path}"
                        }]
                    }
                    logger.error(f"File missing: {funding_path}")
                else:
                    df = pd.read_csv(funding_path, index_col=0)
                    df.index = pd.to_datetime(df.index, format='ISO8601', utc=True)
                    label  = f"{exchange_name}|{asset}|FundingRate"
                    report = self.validate_funding_rates(df, label)
                    if report["passed"]:
                        passed += 1

                all_reports[funding_file] = report

        logger.info(
            f"validate_all complete | "
            f"{passed}/{total} files passed"
        )

        return all_reports

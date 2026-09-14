import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def setup_logging(config: dict) -> None:

    log_cfg   = config.get("logging", {})
    log_level = getattr(logging, log_cfg.get("level", "INFO"))
    log_dir   = config.get("paths", {}).get("log_dir", "logs/")

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"run_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)   

    formatter = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    if log_cfg.get("log_to_file", True):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.addHandler(console_handler)

    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(
        f"Logging configured | "
        f"level={log_cfg.get('level', 'INFO')} | "
        f"file={log_file}"
    )


def load_config(config_path: str = "config/config.yaml") -> dict:

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected at: {os.path.abspath(config_path)}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_sections = [
        "project", "market_data", "cointegration",
        "kalman", "bocpd", "signals", "costs",
        "portfolio", "backtest", "evaluation",
        "paths", "logging"
    ]

    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(
            f"Config missing required sections: {missing}"
        )

    import numpy as np
    import random
    seed = config["project"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)

    logging.info(
        f"Config loaded | "
        f"project={config['project']['name']} | "
        f"version={config['project']['version']} | "
        f"seed={seed}"
    )

    return config


def load_data(config: dict) -> tuple:

    from src.data.fetcher import DataFetcher
    from src.data.validator import DataValidator
    from src.data.preprocessor import DataPreprocessor

    logger = logging.getLogger(__name__)

    processed_dir = config["paths"]["processed_data_dir"]
    assets        = config["market_data"]["assets"]
    exchanges     = config["market_data"]["exchanges"]
    interval      = config["market_data"]["interval"]

    expected_files = [
        os.path.join(
            processed_dir,
            f"{ex}_{a.replace('-', '')}_{interval}_processed.csv"
        )
        for ex in exchanges
        for a in assets
    ]

    # all_exist = all(os.path.exists(f) for f in expected_files)

    # if all_exist:
    #     logger.info(
    #         "Processed data found on disk — skipping fetch. "
    #         "Delete data/processed/ to re-fetch."
    #     )
    # else:
    #     logger.info("Processed data not found — running full pipeline...")

    #     logger.info("Step 1/3: Fetching raw data...")
    #     # fetcher = DataFetcher(config)
    #     # fetcher.fetch_all()

    #     logger.info("Step 2/3: Validating raw data...")
    #     validator = DataValidator(config)
    #     reports   = validator.validate_all()

    #     failed = [
    #         name for name, r in reports.items()
    #         if not r["passed"]
    #     ]

    #     if failed:
    #         logger.warning(
    #             f"Validation failed for {len(failed)} files: {failed}\n"
    #             f"Proceeding with available data. "
    #             f"Check logs for details."
    #         )

    #     logger.info("Step 3/3: Preprocessing data...")
    #     preprocessor = DataPreprocessor(config)
    #     preprocessor.preprocess_all()

    logger.info("Loading processed data into memory...")
    preprocessor = DataPreprocessor(config)
    ohlcv_dict, funding_dict = _load_processed_files(
        config, processed_dir, assets, exchanges, interval
    )

    logger.info(
        f"Data loaded | "
        f"{len(ohlcv_dict)} OHLCV datasets | "
        f"{len(funding_dict)} funding datasets"
    )

    return ohlcv_dict, funding_dict


def _load_processed_files(
    config:        dict,
    processed_dir: str,
    assets:        list,
    exchanges:     list,
    interval:      str
) -> tuple:

    import pandas as pd

    ohlcv_dict   = {}
    funding_dict = {}

    for exchange in exchanges:
        for asset in assets:
            safe  = asset.replace("-", "")
            key   = f"{exchange}_{safe}"

            ohlcv_path = os.path.join(
                processed_dir,
                f"{key}_{interval}_processed.csv"
            )
            funding_path = os.path.join(
                processed_dir,
                f"{key}_funding_processed.csv"
            )

            if os.path.exists(ohlcv_path):
                ohlcv_dict[key] = pd.read_csv(
                    ohlcv_path,
                    index_col=0,
                    parse_dates=True
                )

            if os.path.exists(funding_path):
                funding_dict[key] = pd.read_csv(
                    funding_path,
                    index_col=0,
                    parse_dates=True
                )

    return ohlcv_dict, funding_dict


def build_asset_keys(config: dict) -> list:

    assets    = config["market_data"]["assets"]
    exchanges = config["market_data"]["exchanges"]

    # Primary exchange is first in config list
    primary_exchange = exchanges[0]

    return [
        f"{primary_exchange}_{a.replace('-', '')}"
        for a in assets
    ]


def run_backtest(
    config:       dict,
    ohlcv_dict:   dict,
    funding_dict: dict,
    asset_keys:   list
) -> dict:
 
    from src.backtest.engine import BacktestEngine

    logger = logging.getLogger(__name__)
    logger.info("Starting multi-case backtest...")

    engine  = BacktestEngine(config)
    results = engine.run(ohlcv_dict, funding_dict)

    logger.info(
        f"Backtest complete | "
        f"final_equity={results.get('final_equity', 0):,.0f}"
    )

    return results

def run_backtest(
    config:       dict,
    ohlcv_dict:   dict,
    funding_dict: dict,
) -> dict:
    """Run the main walk-forward backtest across all cases."""
    from src.backtest.engine import BacktestEngine

    logger = logging.getLogger(__name__)


    engine  = BacktestEngine(config)
    results = engine.run(ohlcv_dict, funding_dict)
    # ← no asset_keys parameter

    logger.info(
        f"Backtest complete | "
        f"final_equity={results.get('final_equity', 0):,.0f}"
    )

    return results


# And update the call in main():

# ← removed asset_keys argument


def run_evaluation(
    config:        dict,
    results:       dict,
    ohlcv_dict:    dict,
    funding_dict:  dict,
    asset_keys:    list
) -> dict:
    """
    Run the full evaluation suite:
    1. Compute strategy performance metrics
    2. Run benchmark strategies
    3. Compare strategy vs benchmarks
    4. Run Monte Carlo permutation test
    5. Return complete evaluation report
    """
    from src.backtest.metrics import MetricsCalculator
    from src.evaluation.benchmark import BenchmarkRunner
    from src.evaluation.permutation_test import PermutationTest

    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation suite...")

    metrics    = MetricsCalculator(config)
    benchmarks = BenchmarkRunner(config)
    perm_test  = PermutationTest(config)

    # Step 1: strategy metrics
    logger.info("Computing strategy metrics...")
    strategy_report = metrics.compute(results)

    # Step 2: static OU baseline
    logger.info("Running Static OU benchmark...")
    static_ou_report = benchmarks.run_static_ou(
        ohlcv_dict, funding_dict, asset_keys
    )

    # Step 3: momentum benchmark
    logger.info("Running Momentum benchmark...")
    momentum_report = benchmarks.run_momentum(
        ohlcv_dict, asset_keys
    )

    # Step 4: comparison tables
    logger.info("Comparing strategy to benchmarks...")
    vs_static   = metrics.compare_to_benchmark(
        strategy_report, static_ou_report, "Static OU"
    )
    vs_momentum = metrics.compare_to_benchmark(
        strategy_report, momentum_report, "Momentum"
    )

    # Step 5: permutation test
    logger.info("Running Monte Carlo permutation test...")
    equity_curve = results.get(
        "equity_curve", __import__("pandas").Series()
    )
    perm_results = perm_test.run_both_metrics(equity_curve)

    logger.info(
        f"Permutation test | "
        f"Sharpe p={perm_results['sharpe'].get('p_value', 1):.4f} | "
        f"Calmar p={perm_results['calmar'].get('p_value', 1):.4f} | "
        f"significant={perm_results['both_significant']}"
    )

    evaluation = {
        "strategy":          strategy_report,
        "static_ou":         static_ou_report,
        "momentum":          momentum_report,
        "vs_static_ou":      vs_static,
        "vs_momentum":       vs_momentum,
        "permutation_test":  perm_results,
    }

    return evaluation


def save_results(
    results:    dict,
    evaluation: dict,
    config:     dict
) -> None:
    """
    Save backtest results and evaluation report to disk.

    Saves three files:
    1. equity_curve.csv    — minute-level equity over time
    2. trade_history.csv   — every trade with full metadata
    3. evaluation.yaml     — human-readable performance report

    Why YAML for the evaluation report and not JSON?
    YAML is human-readable without a viewer.
    A portfolio manager can open it in any text editor
    and read the Sharpe ratio without writing any code.
    JSON requires a formatter to be readable.
    """
    import pandas as pd
    import json

    logger      = logging.getLogger(__name__)
    results_dir = config["paths"]["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Equity curve
    equity = results.get("equity_curve", pd.Series())
    if not equity.empty:
        path = os.path.join(
            results_dir, f"equity_curve_{timestamp}.csv"
        )
        equity.to_csv(path)
        logger.info(f"Equity curve saved → {path}")

    # Trade history
    trades = results.get("trade_history", pd.DataFrame())
    if not trades.empty:
        path = os.path.join(
            results_dir, f"trade_history_{timestamp}.csv"
        )
        trades.to_csv(path)
        logger.info(f"Trade history saved → {path}")

    # Evaluation report — strip numpy arrays before saving
    def _serialise(obj):
        """Convert non-serialisable types for JSON/YAML output."""
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialise(v) for v in obj]
        return obj

    clean_evaluation = _serialise(evaluation)
    path = os.path.join(
        results_dir, f"evaluation_{timestamp}.json"
    )
    with open(path, "w") as f:
        json.dump(clean_evaluation, f, indent=2, default=str)
    logger.info(f"Evaluation report saved → {path}")


def main():
    """
    Entry point for the entire system.

    Execution order is strict and load-bearing:

    1. Load config      — everything else needs it
    2. Setup logging    — must happen before any module init
    3. Load data        — fetch, validate, preprocess
    4. Run backtest     — walk-forward engine
    5. Run evaluation   — metrics, benchmarks, permutation test
    6. Save results     — persist to disk

    Each step logs its start and completion.
    If any step raises an exception, the logging system
    captures the full traceback to the log file for diagnosis.
    """
    try:
        config = load_config(str(DEFAULT_CONFIG_PATH))
        setup_logging(config)

        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(
            f"Starting {config['project']['name']} "
            f"v{config['project']['version']}"
        )
        logger.info("=" * 60)

        # Step 3: data pipeline
        ohlcv_dict, funding_dict = load_data(config)
        asset_keys = build_asset_keys(config)

        if not ohlcv_dict:
            logger.error("No data loaded — aborting")
            sys.exit(1)

        # Step 4: backtest
        results = run_backtest(config, ohlcv_dict, funding_dict)

        if not results:
            logger.error("Backtest produced no results — aborting")
            sys.exit(1)

        # Step 5: evaluation
        evaluation = run_evaluation(
            config, results,
            ohlcv_dict, funding_dict, asset_keys
        )

        # Step 6: save
        save_results(results, evaluation, config)

        logger.info("=" * 60)
        logger.info("Pipeline complete. Check results/ for output.")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        # Config or data file missing — user error, clear message
        print(f"ERROR: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        logging.getLogger(__name__).info(
            "Run interrupted by user"
        )
        sys.exit(0)

    except Exception as e:
        # Unexpected error — log full traceback for debugging
        logging.getLogger(__name__).exception(
            f"Unexpected error: {e}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

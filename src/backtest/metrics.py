# src/backtest/metrics.py

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


class MetricsCalculator:

    MINUTES_PER_YEAR = 252 * 1440

    def __init__(self, config: dict):
        self.config      = config
        self.eval_cfg    = config["evaluation"]
        self.risk_free   = self.eval_cfg["risk_free_rate"]

        logger.info(
            f"MetricsCalculator initialized | "
            f"risk_free_rate={self.risk_free:.2%}"
        )

    def _equity_to_returns(
        self,
        equity_curve: pd.Series
    ) -> pd.Series:
        
        log_returns = np.log(equity_curve / equity_curve.shift(1))
        return log_returns.dropna()

    def _annualise_return(
        self,
        period_return: float,
        n_periods: int
    ) -> float:
  
        if n_periods <= 0:
            return 0.0

        scale = self.MINUTES_PER_YEAR / n_periods
        return period_return * scale

    def _annualise_volatility(
        self,
        return_std: float,
        n_periods: int = 1
    ) -> float:

        return return_std * np.sqrt(self.MINUTES_PER_YEAR)

    def _sharpe_ratio(
        self,
        returns: pd.Series
    ) -> float:

        if returns.empty or returns.std() == 0:
            return 0.0

        rf_per_minute = self.risk_free / self.MINUTES_PER_YEAR
        excess        = returns - rf_per_minute

        sharpe = (
            excess.mean() / excess.std()
        ) * np.sqrt(self.MINUTES_PER_YEAR)

        return float(sharpe)

    def _calmar_ratio(
        self,
        annualised_return: float,
        max_drawdown: float
    ) -> float:

        if max_drawdown <= 0:
            return np.inf if annualised_return > 0 else 0.0

        return float(annualised_return / max_drawdown)

    def _max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> Dict:

        if equity_curve.empty:
            return {"max_drawdown": 0.0, "max_duration_days": 0}

        hwm      = equity_curve.cummax()
        drawdown = (hwm - equity_curve) / hwm
        max_dd   = float(drawdown.max())

        below_hwm    = drawdown > 0
        duration_days = 0

        if below_hwm.any():
            groups = (below_hwm != below_hwm.shift()).cumsum()
            underwater_groups = groups[below_hwm]

            if not underwater_groups.empty:
                longest_group = underwater_groups.value_counts().idxmax()
                longest_mask  = (groups == longest_group) & below_hwm
                duration_minutes = longest_mask.sum()
                duration_days    = duration_minutes / 1440

        return {
            "max_drawdown":      max_dd,
            "max_duration_days": round(duration_days, 1),
            "drawdown_series":   drawdown,
        }

    def _turnover_metrics(
        self,
        trade_history: pd.DataFrame,
        initial_capital: float
    ) -> Dict:

        if trade_history.empty:
            return {
                "annual_turnover":        0.0,
                "total_costs":            0.0,
                "turnover_adjusted_return": 0.0,
                "cost_drag_pct":          0.0,
            }

        total_notional = (
            trade_history["size_usd"] * 2
        ).sum()    # both legs

        total_costs = trade_history["close_cost"].sum() + \
                      trade_history["funding_paid"].sum()

        if len(trade_history) > 0:
            first = trade_history["entry_time"].min()
            last  = trade_history["exit_time"].max()
            years = max(
                (last - first).total_seconds() / (365.25 * 86400),
                1 / 365.25
            )
        else:
            years = 1.0

        annual_turnover = total_notional / (initial_capital * years)

        total_net_pnl   = trade_history["net_pnl"].sum()
        gross_pnl       = trade_history["gross_pnl"].sum()

        cost_drag_pct = (
            (gross_pnl - total_net_pnl) / abs(gross_pnl)
            if gross_pnl != 0 else 0.0
        )

        return {
            "annual_turnover":          round(annual_turnover, 2),
            "total_costs":              round(total_costs, 2),
            "turnover_adjusted_return": round(total_net_pnl, 2),
            "cost_drag_pct":            round(cost_drag_pct, 4),
        }

    def _trade_statistics(
        self,
        trade_history: pd.DataFrame
    ) -> Dict:
   
        if trade_history.empty:
            return {}

        winners = trade_history[trade_history["net_pnl"] > 0]
        losers  = trade_history[trade_history["net_pnl"] <= 0]

        gross_wins   = winners["net_pnl"].sum() if not winners.empty else 0
        gross_losses = abs(losers["net_pnl"].sum()) if not losers.empty else 1

        profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

        holding_days = trade_history["holding_days"]
        close_reasons = trade_history["close_reason"].value_counts()
        close_reason_pct = (
            close_reasons / len(trade_history) * 100
        ).round(1).to_dict()

        return {
            "n_trades":           len(trade_history),
            "win_rate":           round(len(winners) / len(trade_history), 4),
            "profit_factor":      round(profit_factor, 4),
            "avg_holding_days":   round(holding_days.mean(), 2),
            "median_holding_days": round(holding_days.median(), 2),
            "max_holding_days":   round(holding_days.max(), 2),
            "avg_win_pnl":        round(winners["net_pnl"].mean(), 2)
                                  if not winners.empty else 0,
            "avg_loss_pnl":       round(losers["net_pnl"].mean(), 2)
                                  if not losers.empty else 0,
            "close_reason_pct":   close_reason_pct,
            "long_trades_pct":    round(
                (trade_history["direction"] == 1).mean(), 4
            ),
        }

    def _distribution_statistics(
        self,
        returns: pd.Series
    ) -> Dict:

        if returns.empty or len(returns) < 4:
            return {}

        return {
            "skewness":     round(float(stats.skew(returns)), 4),
            "kurtosis":     round(float(stats.kurtosis(returns)), 4),
            "jarque_bera_p": round(
                float(stats.jarque_bera(returns).pvalue), 6
            ),
            "var_95":       round(
                float(np.percentile(returns, 5)), 6
            ),
            "cvar_95":      round(
                float(returns[returns <= np.percentile(returns, 5)].mean()),
                6
            ),
        }

    def compute(
        self,
        results: Dict
    ) -> Dict:

        equity_curve  = results.get("equity_curve", pd.Series())
        trade_history = results.get("trade_history", pd.DataFrame())

        if equity_curve.empty:
            logger.error("Empty equity curve — cannot compute metrics")
            return {}

        initial_capital = self.config["portfolio"]["initial_capital"]
        returns         = self._equity_to_returns(equity_curve)
        n_periods       = len(returns)

        total_log_return  = returns.sum()
        annual_return     = self._annualise_return(
            total_log_return, n_periods
        )
        annual_vol        = self._annualise_volatility(returns.std())
        sharpe            = self._sharpe_ratio(returns)

        dd_result  = self._max_drawdown(equity_curve)
        max_dd     = dd_result["max_drawdown"]
        calmar     = self._calmar_ratio(annual_return, max_dd)

        trade_stats    = self._trade_statistics(trade_history)
        turnover_stats = self._turnover_metrics(
            trade_history, initial_capital
        )

        dist_stats = self._distribution_statistics(returns)

        report = {
            "returns": {
                "annualised_return":    round(annual_return, 4),
                "annualised_vol":       round(annual_vol, 4),
                "total_return":         round(
                    np.exp(total_log_return) - 1, 4
                ),
                "sharpe_ratio":         round(sharpe, 4),
                "calmar_ratio":         round(calmar, 4),
                "final_equity":         round(
                    equity_curve.iloc[-1], 2
                ),
            },
            "risk": {
                "max_drawdown":         round(max_dd, 4),
                "max_drawdown_duration_days":
                                        dd_result["max_duration_days"],
                "var_95":               dist_stats.get("var_95", 0),
                "cvar_95":              dist_stats.get("cvar_95", 0),
            },
            "trades":   trade_stats,
            "costs":    turnover_stats,
            "distribution": {
                "skewness":     dist_stats.get("skewness", 0),
                "kurtosis":     dist_stats.get("kurtosis", 0),
                "jarque_bera_p": dist_stats.get("jarque_bera_p", 1),
                "is_normal":    dist_stats.get(
                    "jarque_bera_p", 0
                ) > 0.05,
            },
            "meta": {
                "n_windows":     results.get("n_windows", 0),
                "n_windows_ok":  results.get("n_windows_ok", 0),
                "n_trades":      len(trade_history),
                "backtest_days": n_periods / 1440,
            }
        }

        self._log_summary(report)
        return report

    def _log_summary(self, report: Dict) -> None:

        r = report.get("returns", {})
        k = report.get("risk", {})
        t = report.get("trades", {})
        c = report.get("costs", {})
        d = report.get("distribution", {})

        logger.info("=" * 60)
        logger.info("BACKTEST PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"Annualised Return : {r.get('annualised_return', 0):.2%}"
        )
        logger.info(
            f"Annualised Vol    : {r.get('annualised_vol', 0):.2%}"
        )
        logger.info(
            f"Sharpe Ratio      : {r.get('sharpe_ratio', 0):.3f}"
        )
        logger.info(
            f"Calmar Ratio      : {r.get('calmar_ratio', 0):.3f}"
        )
        logger.info(
            f"Max Drawdown      : {k.get('max_drawdown', 0):.2%}"
        )
        logger.info(
            f"Drawdown Duration : {k.get('max_drawdown_duration_days', 0):.0f}d"
        )
        logger.info(
            f"Win Rate          : {t.get('win_rate', 0):.1%}"
        )
        logger.info(
            f"Profit Factor     : {t.get('profit_factor', 0):.2f}"
        )
        logger.info(
            f"Avg Holding       : {t.get('avg_holding_days', 0):.2f}d"
        )
        logger.info(
            f"Annual Turnover   : {c.get('annual_turnover', 0):.1f}x"
        )
        logger.info(
            f"Cost Drag         : {c.get('cost_drag_pct', 0):.2%}"
        )
        logger.info(
            f"Skewness          : {d.get('skewness', 0):.3f}"
        )
        logger.info(
            f"Kurtosis          : {d.get('kurtosis', 0):.3f}"
        )
        logger.info(
            f"Returns Normal?   : {d.get('is_normal', False)}"
        )
        logger.info("=" * 60)

    def compare_to_benchmark(
        self,
        strategy_report: Dict,
        benchmark_report: Dict,
        benchmark_name: str = "Static OU"
    ) -> Dict:

        def _get(report, *keys):
            val = report
            for k in keys:
                val = val.get(k, {})
            return val if not isinstance(val, dict) else 0.0

        comparison = {
            "metric": [
                "Annualised Return",
                "Annualised Vol",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "Max DD Duration (days)",
                "Win Rate",
                "Profit Factor",
                "Avg Holding (days)",
                "Annual Turnover",
                "Cost Drag",
                "Skewness",
                "Kurtosis",
            ],
            "strategy": [
                _get(strategy_report,  "returns", "annualised_return"),
                _get(strategy_report,  "returns", "annualised_vol"),
                _get(strategy_report,  "returns", "sharpe_ratio"),
                _get(strategy_report,  "returns", "calmar_ratio"),
                _get(strategy_report,  "risk",    "max_drawdown"),
                _get(strategy_report,  "risk",    "max_drawdown_duration_days"),
                _get(strategy_report,  "trades",  "win_rate"),
                _get(strategy_report,  "trades",  "profit_factor"),
                _get(strategy_report,  "trades",  "avg_holding_days"),
                _get(strategy_report,  "costs",   "annual_turnover"),
                _get(strategy_report,  "costs",   "cost_drag_pct"),
                _get(strategy_report,  "distribution", "skewness"),
                _get(strategy_report,  "distribution", "kurtosis"),
            ],
            benchmark_name: [
                _get(benchmark_report, "returns", "annualised_return"),
                _get(benchmark_report, "returns", "annualised_vol"),
                _get(benchmark_report, "returns", "sharpe_ratio"),
                _get(benchmark_report, "returns", "calmar_ratio"),
                _get(benchmark_report, "risk",    "max_drawdown"),
                _get(benchmark_report, "risk",    "max_drawdown_duration_days"),
                _get(benchmark_report, "trades",  "win_rate"),
                _get(benchmark_report, "trades",  "profit_factor"),
                _get(benchmark_report, "trades",  "avg_holding_days"),
                _get(benchmark_report, "costs",   "annual_turnover"),
                _get(benchmark_report, "costs",   "cost_drag_pct"),
                _get(benchmark_report, "distribution", "skewness"),
                _get(benchmark_report, "distribution", "kurtosis"),
            ],
        }

        df = pd.DataFrame(comparison).set_index("metric")
        logger.info(f"\nStrategy vs {benchmark_name}:\n{df.to_string()}")

        return comparison

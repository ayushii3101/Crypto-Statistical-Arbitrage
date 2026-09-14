import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class Position:

    position_id:      str
    entry_time:       datetime
    entry_zscore:     float
    entry_spread:     float
    direction:        int           # +1 = long spread, -1 = short spread
    size_usd:         float         # notional per leg
    entry_threshold:  float         # z-score threshold at entry
    exit_threshold:   float         # z-score threshold at exit
    half_life_days:   float         # OU half-life at entry
    round_trip_cost:  float         # estimated cost at entry
    max_holding_days: int           # forced exit after this many days
    pnl:              float = 0.0   # running mark-to-market PnL
    funding_paid:     float = 0.0   # cumulative funding cost paid


@dataclass
class PortfolioState:

    cash:             float
    open_positions:   List[Position] = field(default_factory=list)
    closed_trades:    List[dict]     = field(default_factory=list)
    peak_equity:      float          = 0.0
    total_pnl:        float          = 0.0
    n_trades:         int            = 0

    @property
    def equity(self) -> float:
  
        open_pnl = sum(p.pnl for p in self.open_positions)
        return self.cash + open_pnl

    @property
    def deployed_capital(self) -> float:
  
        return sum(p.size_usd * 2 for p in self.open_positions)

    @property
    def current_drawdown(self) -> float:
  
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.equity) / self.peak_equity
        return max(dd, 0.0)    # floor at zero


class PortfolioManager:
  
    def __init__(self, config: dict):
        self.config   = config
        self.port_cfg = config["portfolio"]
        self.sig_cfg  = config["signals"]

        self.initial_capital   = self.port_cfg["initial_capital"]
        self.max_position_pct  = self.port_cfg["max_position_pct"]
        self.max_drawdown_pct  = self.port_cfg["max_drawdown_pct"]
        self.max_open_trades   = self.port_cfg["max_open_trades"]
        self.max_holding_days  = self.sig_cfg["max_holding_period_days"]

        logger.info(
            f"PortfolioManager initialized | "
            f"capital={self.initial_capital:,} | "
            f"max_pos_pct={self.max_position_pct:.0%} | "
            f"max_dd={self.max_drawdown_pct:.0%} | "
            f"max_open={self.max_open_trades}"
        )

    def _base_position_size(
        self,
        equity: float
    ) -> float:
    
        return equity * self.max_position_pct

    def _volatility_scalar(
        self,
        spread_std: float,
        target_vol: float = 0.02
    ) -> float:

        if spread_std <= 0 or np.isnan(spread_std):
            return 0.5    

        scalar = target_vol / spread_std
        scalar = np.clip(scalar, 0.1, 1.0)

        logger.debug(
            f"Vol scalar | spread_std={spread_std:.4f} | "
            f"target_vol={target_vol:.4f} | scalar={scalar:.4f}"
        )

        return scalar

    def _zscore_scalar(
        self,
        zscore: float,
        entry_threshold: float
    ) -> float:

        if entry_threshold <= 0:
            return 0.5

        abs_z  = abs(zscore)
        ratio  = abs_z / entry_threshold
        scalar = np.clip(ratio / 2.0, 0.5, 1.0)

        logger.debug(
            f"Z-score scalar | z={zscore:.4f} | "
            f"threshold={entry_threshold:.4f} | scalar={scalar:.4f}"
        )

        return scalar

    def _drawdown_scalar(
        self,
        current_drawdown: float
    ) -> float:

        max_dd = self.max_drawdown_pct
        if current_drawdown >= max_dd:
            logger.warning(
                f"Max drawdown reached: {current_drawdown:.1%} >= "
                f"{max_dd:.1%}. Blocking all new positions."
            )
            return 0.0

        scalar = 1.0 - (current_drawdown / max_dd)
        scalar = np.clip(scalar, 0.0, 1.0)

        if current_drawdown > 0.05:
            logger.warning(
                f"Drawdown={current_drawdown:.1%} | "
                f"size_scalar={scalar:.2f}"
            )

        return scalar

    def _capacity_scalar(
        self,
        state: PortfolioState
    ) -> float:

        n_open = len(state.open_positions)
        if n_open >= self.max_open_trades:
            logger.debug(
                f"Max open trades reached: {n_open}/{self.max_open_trades}"
            )
            return 0.0

        min_trade_size = self.initial_capital * 0.01    # 1% floor
        available_cash = state.cash

        if available_cash < min_trade_size:
            logger.warning(
                f"Insufficient cash: {available_cash:.2f} < "
                f"{min_trade_size:.2f}"
            )
            return 0.0

        return 1.0

    def compute_position_size(
        self,
        state: PortfolioState,
        zscore: float,
        entry_threshold: float,
        spread_std: float,
        is_tradeable: bool
    ) -> float:
       
        if not is_tradeable:
            logger.debug("Position size = 0: not tradeable")
            return 0.0

        equity = state.equity

        if equity <= 0:
            logger.warning("Equity <= 0, cannot size position")
            return 0.0
        if equity > state.peak_equity:
            state.peak_equity = equity

        base = self._base_position_size(equity)
        vol_scalar = self._volatility_scalar(spread_std)
        z_scalar   = self._zscore_scalar(zscore, entry_threshold)
        dd_scalar  = self._drawdown_scalar(state.current_drawdown)
        cap_scalar = self._capacity_scalar(state)

        size = base * vol_scalar * z_scalar * dd_scalar * cap_scalar
        min_size = self.initial_capital * 0.01    
        if size < min_size and cap_scalar > 0:
            logger.debug(
                f"Computed size {size:.2f} below minimum {min_size:.2f}"
                f" — rounding to zero"
            )
            size = 0.0

        logger.info(
            f"Position size | "
            f"base={base:.0f} | "
            f"vol_s={vol_scalar:.2f} | "
            f"z_s={z_scalar:.2f} | "
            f"dd_s={dd_scalar:.2f} | "
            f"cap_s={cap_scalar:.2f} | "
            f"final={size:.0f}"
        )

        return size

    def open_position(
        self,
        state: PortfolioState,
        timestamp: datetime,
        zscore: float,
        spread_value: float,
        size_usd: float,
        entry_threshold: float,
        exit_threshold: float,
        half_life_days: float,
        round_trip_cost: float,
        position_id: Optional[str] = None
    ) -> Optional[Position]:

        if size_usd <= 0:
            return None

        direction = -1 if zscore > 0 else 1
        margin_per_leg = size_usd * 0.10
        total_margin   = margin_per_leg * 2  

        if total_margin > state.cash:
            logger.warning(
                f"Insufficient cash for margin: "
                f"need {total_margin:.0f}, have {state.cash:.0f}"
            )
            return None

        pid = position_id or f"pos_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        position = Position(
            position_id     = pid,
            entry_time      = timestamp,
            entry_zscore    = zscore,
            entry_spread    = spread_value,
            direction       = direction,
            size_usd        = size_usd,
            entry_threshold = entry_threshold,
            exit_threshold  = exit_threshold,
            half_life_days  = half_life_days,
            round_trip_cost = round_trip_cost,
            max_holding_days = self.max_holding_days,
        )

        state.open_positions.append(position)
        state.cash -= total_margin
        state.n_trades += 1

        logger.info(
            f"Position opened | {pid} | "
            f"direction={direction} | "
            f"z={zscore:.4f} | size={size_usd:.0f} | "
            f"cash_remaining={state.cash:.0f}"
        )

        return position

    def update_position_pnl(
        self,
        position: Position,
        current_spread: float,
        current_funding_cost: float = 0.0
    ) -> None:

        spread_move = current_spread - position.entry_spread
        gross_pnl   = position.direction * spread_move * position.size_usd

        position.funding_paid += current_funding_cost
        position.pnl           = gross_pnl - position.funding_paid

    def close_position(
        self,
        state: PortfolioState,
        position: Position,
        timestamp: datetime,
        current_spread: float,
        current_zscore: float,
        close_reason: str
    ) -> dict:
    
        self.update_position_pnl(position, current_spread)
        close_cost = position.round_trip_cost * position.size_usd * 0.5
        net_pnl    = position.pnl - close_cost

        margin_returned = position.size_usd * 0.10 * 2
        state.cash     += margin_returned + net_pnl

        holding_days = (
            timestamp - position.entry_time
        ).total_seconds() / 86_400

        trade_record = {
            "position_id":    position.position_id,
            "entry_time":     position.entry_time,
            "exit_time":      timestamp,
            "holding_days":   holding_days,
            "direction":      position.direction,
            "size_usd":       position.size_usd,
            "entry_zscore":   position.entry_zscore,
            "exit_zscore":    current_zscore,
            "entry_spread":   position.entry_spread,
            "exit_spread":    current_spread,
            "gross_pnl":      position.pnl,
            "funding_paid":   position.funding_paid,
            "close_cost":     close_cost,
            "net_pnl":        net_pnl,
            "return_pct":     net_pnl / (position.size_usd * 2),
            "close_reason":   close_reason,
        }

        state.closed_trades.append(trade_record)
        state.open_positions.remove(position)
        state.total_pnl += net_pnl

        logger.info(
            f"Position closed | {position.position_id} | "
            f"reason={close_reason} | "
            f"holding={holding_days:.2f}d | "
            f"net_pnl={net_pnl:.2f} | "
            f"return={net_pnl/(position.size_usd*2):.2%}"
        )

        return trade_record

    def check_forced_exits(
        self,
        state: PortfolioState,
        timestamp: datetime,
        current_spread: float,
        current_zscore: float,
        bocpd_cp_prob: float = 0.0
    ) -> List[dict]:
 
        closed = []
        bocpd_threshold = self.config["bocpd"]["threshold_probability"]

        for position in list(state.open_positions):

            holding_days = (
                timestamp - position.entry_time
            ).total_seconds() / 86_400

            close_reason = None
            if holding_days >= position.max_holding_days:
                close_reason = "max_holding"

            elif bocpd_cp_prob >= bocpd_threshold:
                close_reason = "bocpd_break"

            elif (
                position.direction == -1 and
                current_zscore < -3 * position.entry_threshold
            ) or (
                position.direction == 1 and
                current_zscore > 3 * position.entry_threshold
            ):
                close_reason = "stop_loss"

            if close_reason:
                trade = self.close_position(
                    state, position, timestamp,
                    current_spread, current_zscore,
                    close_reason
                )
                closed.append(trade)

        return closed

    def initialise_state(self) -> PortfolioState:

        state = PortfolioState(
            cash         = self.initial_capital,
            peak_equity  = self.initial_capital,
        )

        logger.info(
            f"Portfolio state initialised | "
            f"capital={self.initial_capital:,}"
        )

        return state

    def get_trade_history(
        self,
        state: PortfolioState
    ) -> pd.DataFrame:

        if not state.closed_trades:
            logger.warning("No closed trades in history")
            return pd.DataFrame()

        df = pd.DataFrame(state.closed_trades)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"]  = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time").reset_index(drop=True)

        logger.info(
            f"Trade history | {len(df)} trades | "
            f"total_pnl={df['net_pnl'].sum():.2f} | "
            f"win_rate={( df['net_pnl'] > 0).mean():.1%}"
        )

        return df

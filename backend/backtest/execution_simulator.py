"""
Execution Simulator for Backtest

Simulates order execution with realistic conditions:
- Slippage calculation
- Fee calculation
- TP/SL order checking
- Position management
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from config.fees import FeeSchedule

from .virtual_account import VirtualAccount, VirtualPosition
from .models import BacktestTradeRecord

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""
    success: bool
    executed_price: float = 0.0
    executed_size: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    error: Optional[str] = None


class ExecutionSimulator:
    """
    Simulates order execution for backtesting.

    Handles:
    - Slippage calculation based on order side
    - Fee calculation (maker/taker)
    - TP/SL order trigger checking
    - Position opening/closing
    """

    def __init__(
        self,
        slippage_percent: Optional[float] = None,
        fee_rate: Optional[float] = None,
        maker_fee_rate: Optional[float] = None,
        realistic_tpsl: bool = True,
        fee_schedule: Optional[FeeSchedule] = None,
    ):
        """
        Initialize execution simulator.

        Args:
            slippage_percent: Default slippage in percent (legacy; converted
                once here via ``FeeSchedule.from_percent``).
            fee_rate: Trading fee rate in percent, taker (legacy).
            maker_fee_rate: Maker fee rate in percent, TP limit (legacy).
            realistic_tpsl: TP needs close-confirm, SL pays taker (default True)
            fee_schedule: Single source of truth. When given, legacy percent
                args must be None; the schedule's bps values are used as-is.
        """
        if fee_schedule is not None:
            if slippage_percent is not None or fee_rate is not None or maker_fee_rate is not None:
                raise ValueError("pass either fee_schedule or legacy percent args, not both")
            self.fees = fee_schedule
        else:
            self.fees = FeeSchedule.from_percent(
                maker_fee_pct=maker_fee_rate if maker_fee_rate is not None else 0.02,
                taker_fee_pct=fee_rate if fee_rate is not None else 0.035,
                slippage_pct=slippage_percent if slippage_percent is not None else 0.05,
            )
        self.realistic_tpsl = realistic_tpsl

    @property
    def slippage_percent(self) -> float:
        """Legacy percent-unit view (``percent = bps / 100``)."""
        return self.fees.slippage_bps / 100.0

    @property
    def fee_rate(self) -> float:
        """Legacy percent-unit taker view."""
        return self.fees.taker_fee_bps / 100.0

    @property
    def maker_fee_rate(self) -> float:
        """Legacy percent-unit maker view."""
        return self.fees.maker_fee_bps / 100.0

    def calculate_execution_price(
        self,
        price: float,
        side: str,
        slippage_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate execution price with slippage.

        Args:
            price: Base price
            side: "buy" or "sell"
            slippage_pct: Override slippage percent

        Returns:
            (executed_price, slippage_amount)
        """
        slippage = slippage_pct if slippage_pct is not None else self.slippage_percent

        if side.lower() == "buy":
            # Buying pushes price up
            executed_price = price * (1 + slippage / 100)
        else:
            # Selling pushes price down
            executed_price = price * (1 - slippage / 100)

        slippage_amount = abs(executed_price - price)
        return executed_price, slippage_amount

    def calculate_fee(
        self,
        notional: float,
        fee_rate: Optional[float] = None,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate trading fee.

        Args:
            notional: Trade notional value (size * price)
            fee_rate: Legacy override fee rate in percent (deprecated; when
                given it is converted once here, not at call sites).
            is_maker: Use the schedule's maker rate when no override given.

        Returns:
            Fee amount
        """
        if fee_rate is not None:
            return abs(notional) * fee_rate / 100
        return self.fees.fee(notional, is_maker=is_maker)

    def check_tp_sl_triggers(
        self,
        account: VirtualAccount,
        prices: Dict[str, float],
        timestamp: int,
    ) -> List[BacktestTradeRecord]:
        """
        Check if any TP/SL orders should trigger.

        Each pending order is independent (like Hyperliquid) - when triggered,
        only closes the portion of position that order controls.

        Args:
            account: Virtual account state
            prices: Current prices {symbol: price}
            timestamp: Current timestamp

        Returns:
            List of trade records for triggered orders
        """
        triggered_trades = []
        orders_to_remove = []

        # Check each pending order independently
        for order in account.pending_orders:
            symbol = order.symbol
            if symbol not in prices:
                continue

            pos = account.get_position(symbol)
            if not pos:
                # Position no longer exists, mark order for removal
                orders_to_remove.append(order.order_id)
                continue

            current_price = prices[symbol]
            should_trigger = False

            # Check trigger condition based on order type and position side
            if order.order_type == "take_profit":
                if pos.side == "long" and current_price >= order.trigger_price:
                    should_trigger = True
                elif pos.side == "short" and current_price <= order.trigger_price:
                    should_trigger = True
            elif order.order_type == "stop_loss":
                if pos.side == "long" and current_price <= order.trigger_price:
                    should_trigger = True
                elif pos.side == "short" and current_price >= order.trigger_price:
                    should_trigger = True

            if should_trigger:
                # Apply slippage to exit price
                close_side = "sell" if pos.side == "long" else "buy"
                executed_price, _ = self.calculate_execution_price(order.trigger_price, close_side)

                # Use actual close size (min of order size and remaining position)
                actual_close_size = min(order.size, pos.size)
                if actual_close_size <= 0:
                    orders_to_remove.append(order.order_id)
                    continue

                # Calculate fee for actual close size
                notional = actual_close_size * executed_price
                fee = self.calculate_fee(notional)

                # Partial close position using order's entry price for accurate PnL
                pnl = account.partial_close_position(
                    symbol=symbol,
                    size=actual_close_size,
                    exit_price=executed_price,
                    fee=fee,
                    entry_price=order.entry_price,
                )

                if pnl is not None:
                    # Calculate PnL percent based on actual close size
                    entry_notional = actual_close_size * order.entry_price
                    pnl_percent = (pnl / entry_notional * 100) if entry_notional > 0 else 0

                    exit_reason = "tp" if order.order_type == "take_profit" else "sl"
                    trade = BacktestTradeRecord(
                        timestamp=order.created_at,
                        trigger_type="",
                        symbol=symbol,
                        operation="close",
                        side=pos.side,
                        entry_price=order.entry_price,
                        size=actual_close_size,
                        leverage=pos.leverage,
                        exit_price=executed_price,
                        exit_timestamp=timestamp,
                        exit_reason=exit_reason,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        fee=fee,
                        reason=f"{'Take Profit' if exit_reason == 'tp' else 'Stop Loss'} triggered",
                    )
                    triggered_trades.append(trade)

                # Mark order for removal (it's been executed)
                orders_to_remove.append(order.order_id)

        # Remove triggered/invalid orders
        for order_id in orders_to_remove:
            account.remove_pending_order(order_id)

        return triggered_trades

    def check_tp_sl_with_klines(
        self,
        account: VirtualAccount,
        klines: List[Dict[str, Any]],
        position_side: str,
        data_provider: Any,
    ) -> List[BacktestTradeRecord]:
        """
        Check TP/SL triggers using K-line high/low prices between triggers.

        This provides more accurate TP/SL detection by checking if price
        touched TP/SL levels at any point, not just at trigger timestamps.

        Args:
            account: Virtual account state
            klines: List of klines between last trigger and current trigger
                    Each kline has: timestamp, high, low, close
            position_side: "long" or "short" for the position
            data_provider: Historical data provider for querying prices of all symbols

        Returns:
            List of trade records for triggered orders, in chronological order
        """
        triggered_trades = []
        orders_to_remove = []

        # Process klines in chronological order
        for kline in klines:
            kline_time_ms = kline["timestamp"] * 1000
            high = kline["high"]
            low = kline["low"]

            # Check each pending order
            for order in list(account.pending_orders):
                if order.order_id in orders_to_remove:
                    continue

                symbol = order.symbol
                pos = account.get_position(symbol)
                if not pos:
                    orders_to_remove.append(order.order_id)
                    continue

                should_trigger = False
                trigger_price = order.trigger_price

                # Check trigger condition using kline high/low
                # PRP-P1-2: TP needs close-confirm (影線碰不算成交), SL keeps wick trigger
                close = kline.get("close", 0) or 0
                if order.order_type == "take_profit":
                    if self.realistic_tpsl:
                        if pos.side == "long" and close >= trigger_price:
                            should_trigger = True
                        elif pos.side == "short" and close <= trigger_price:
                            should_trigger = True
                    else:
                        if pos.side == "long" and high >= trigger_price:
                            should_trigger = True
                        elif pos.side == "short" and low <= trigger_price:
                            should_trigger = True
                elif order.order_type == "stop_loss":
                    if pos.side == "long" and low <= trigger_price:
                        should_trigger = True
                    elif pos.side == "short" and high >= trigger_price:
                        should_trigger = True

                if should_trigger:
                    # Execute at trigger price (not kline close)
                    # PRP-P0-1: SL market penalty x3 slip (穿價保證成交但多付滑點)
                    close_side = "sell" if pos.side == "long" else "buy"
                    eff_slip = self.slippage_percent * 3 if order.order_type == "stop_loss" else self.slippage_percent
                    executed_price, _ = self.calculate_execution_price(trigger_price, close_side, slippage_pct=eff_slip)

                    # Use actual close size (min of order size and remaining position)
                    actual_close_size = min(order.size, pos.size)
                    if actual_close_size <= 0:
                        orders_to_remove.append(order.order_id)
                        continue

                    # Calculate fee for actual close size
                    # PRP-P1-2: TP maker fee, SL taker fee
                    notional = actual_close_size * executed_price
                    fee = self.calculate_fee(notional, is_maker=(order.order_type == "take_profit"))

                    # Partial close position
                    pnl = account.partial_close_position(
                        symbol=symbol,
                        size=actual_close_size,
                        exit_price=executed_price,
                        fee=fee,
                        entry_price=order.entry_price,
                    )

                    if pnl is not None:
                        # Get prices for ALL position symbols at kline time for accurate equity
                        # For single symbol: only current symbol price
                        # For multi symbol: query all position symbols at same timestamp
                        kline_prices = {symbol: kline["close"]}
                        for pos_symbol in account.positions:
                            if pos_symbol != symbol:
                                # Query price at kline timestamp for other symbols
                                other_price = data_provider._get_price_at_time(
                                    pos_symbol, kline_time_ms
                                )
                                if other_price:
                                    kline_prices[pos_symbol] = other_price
                        account.update_equity(kline_prices)

                        entry_notional = actual_close_size * order.entry_price
                        pnl_percent = (pnl / entry_notional * 100) if entry_notional > 0 else 0

                        exit_reason = "tp" if order.order_type == "take_profit" else "sl"
                        trade = BacktestTradeRecord(
                            timestamp=order.created_at,
                            trigger_type="",
                            symbol=symbol,
                            operation="close",
                            side=pos.side,
                            entry_price=order.entry_price,
                            size=actual_close_size,
                            leverage=pos.leverage,
                            exit_price=executed_price,
                            exit_timestamp=kline_time_ms,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_percent=pnl_percent,
                            fee=fee,
                            equity_after=account.equity,  # Record equity after this trade
                            reason=f"{'Take Profit' if exit_reason == 'tp' else 'Stop Loss'} triggered",
                        )
                        triggered_trades.append(trade)

                    orders_to_remove.append(order.order_id)

        # Remove triggered orders
        for order_id in orders_to_remove:
            account.remove_pending_order(order_id)

        return triggered_trades

    def execute_decision(
        self,
        decision: Any,
        account: VirtualAccount,
        current_price: float,
        timestamp: int,
        trigger_type: str = "",
        pool_name: Optional[str] = None,
        triggered_signals: Optional[List[str]] = None,
    ) -> Optional[BacktestTradeRecord]:
        """
        Execute a trading decision.

        Args:
            decision: Decision object from strategy
            account: Virtual account state
            current_price: Current market price
            timestamp: Current timestamp
            trigger_type: "signal" or "scheduled"
            pool_name: Signal pool name (if signal trigger)
            triggered_signals: List of triggered signal names

        Returns:
            Trade record if trade executed, None otherwise
        """
        operation = decision.operation.lower() if decision.operation else "hold"

        if operation == "hold":
            return None

        symbol = decision.symbol
        has_position = account.has_position(symbol)

        # Handle close operation
        if operation == "close":
            if not has_position:
                return None
            target_portion = getattr(decision, 'target_portion_of_balance', 1.0)
            return self._execute_close(
                account, symbol, current_price, timestamp,
                trigger_type, pool_name, triggered_signals, decision.reason,
                target_portion=target_portion
            )

        # Handle buy/sell operations
        if operation in ("buy", "sell"):
            if has_position:
                pos = account.get_position(symbol)
                # If same direction, add to position (averaging)
                if (operation == "buy" and pos.side == "long") or \
                   (operation == "sell" and pos.side == "short"):
                    return self._execute_add_position(
                        account, decision, current_price, timestamp,
                        trigger_type, pool_name, triggered_signals
                    )
                # If opposite direction, close existing first (full close)
                # and stash the close trade so the engine can collect both
                close_trade = self._execute_close(
                    account, symbol, current_price, timestamp,
                    trigger_type, pool_name, triggered_signals, "Reverse position",
                    target_portion=1.0
                )
                open_trade = self._execute_open(
                    account, decision, current_price, timestamp,
                    trigger_type, pool_name, triggered_signals
                )
                if close_trade is not None:
                    # stash for engine loop (supports both old and new engine paths)
                    self._last_close_trade = close_trade  # type: ignore
                    # if open failed, still return the close so it is not lost
                    if open_trade is None:
                        return close_trade
                    # return both via a tuple-like container; engine will unpack
                    # keep backward compat: attach close as attribute on open
                    try:
                        open_trade._close_trade = close_trade  # type: ignore
                    except Exception:
                        pass
                return open_trade

            return self._execute_open(
                account, decision, current_price, timestamp,
                trigger_type, pool_name, triggered_signals
            )

        return None

    def _execute_add_position(
        self,
        account: VirtualAccount,
        decision: Any,
        current_price: float,
        timestamp: int,
        trigger_type: str,
        pool_name: Optional[str],
        triggered_signals: Optional[List[str]],
    ) -> Optional[BacktestTradeRecord]:
        """Execute adding to existing position (averaging entry price)."""
        operation = decision.operation.lower()
        symbol = decision.symbol
        side = "long" if operation == "buy" else "short"

        # Calculate execution price with slippage
        exec_price, slippage = self.calculate_execution_price(current_price, operation)

        # Calculate additional position size
        portion = getattr(decision, 'target_portion_of_balance', 0.5)
        leverage = getattr(decision, 'leverage', 1)
        available = account.balance * portion
        add_size = (available * leverage) / exec_price

        if add_size <= 0:
            return None

        # Calculate fee
        notional = add_size * exec_price
        fee = self.calculate_fee(notional)

        # Get TP/SL prices for this specific order
        tp_price = getattr(decision, 'take_profit_price', None)
        sl_price = getattr(decision, 'stop_loss_price', None)

        # Validate TP/SL prices - abort entire backtest if invalid
        if tp_price is not None or sl_price is not None:
            from program_trader.executor import validate_tp_sl_prices
            is_valid, errors = validate_tp_sl_prices(
                operation=operation,
                entry_price=exec_price,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
            )
            if not is_valid:
                raise ValueError(
                    f"Invalid TP/SL in strategy: {'; '.join(errors)}. "
                    f"Please fix your strategy code."
                )

        # Get position info before adding
        pos = account.get_position(symbol)
        old_size = pos.size
        old_entry = pos.entry_price

        # Add to position (no longer pass TP/SL to position itself)
        account.add_to_position(
            symbol=symbol,
            size=add_size,
            entry_price=exec_price,
            fee=fee,
        )

        # Create independent TP/SL orders for this portion
        close_side = "sell" if side == "long" else "buy"
        if tp_price:
            account.add_pending_order(
                symbol=symbol,
                side=close_side,
                order_type="take_profit",
                trigger_price=tp_price,
                size=add_size,
                entry_price=exec_price,
                timestamp=timestamp,
            )
        if sl_price:
            account.add_pending_order(
                symbol=symbol,
                side=close_side,
                order_type="stop_loss",
                trigger_price=sl_price,
                size=add_size,
                entry_price=exec_price,
                timestamp=timestamp,
            )

        # Get updated position info
        updated_pos = account.get_position(symbol)

        return BacktestTradeRecord(
            timestamp=timestamp,
            trigger_type=trigger_type,
            symbol=symbol,
            operation="add_position",
            side=side,
            entry_price=exec_price,
            size=add_size,
            leverage=leverage,
            fee=fee,
            reason=getattr(decision, 'reason', '') + f" (Added to position, avg entry: {updated_pos.entry_price:.2f})",
            pool_name=pool_name,
            triggered_signals=triggered_signals or [],
        )

    def _execute_open(
        self,
        account: VirtualAccount,
        decision: Any,
        current_price: float,
        timestamp: int,
        trigger_type: str,
        pool_name: Optional[str],
        triggered_signals: Optional[List[str]],
    ) -> Optional[BacktestTradeRecord]:
        """Execute position open."""
        operation = decision.operation.lower()
        symbol = decision.symbol
        side = "long" if operation == "buy" else "short"

        # Calculate execution price with slippage
        exec_price, slippage = self.calculate_execution_price(current_price, operation)

        # Calculate position size
        portion = getattr(decision, 'target_portion_of_balance', 0.5)
        leverage = getattr(decision, 'leverage', 1)
        available = account.balance * portion
        size = (available * leverage) / exec_price

        if size <= 0:
            return None

        # Calculate entry fee
        notional = size * exec_price
        fee = self.calculate_fee(notional)

        # Get TP/SL prices
        tp_price = getattr(decision, 'take_profit_price', None)
        sl_price = getattr(decision, 'stop_loss_price', None)

        # Validate TP/SL prices - abort entire backtest if invalid
        if tp_price is not None or sl_price is not None:
            from program_trader.executor import validate_tp_sl_prices
            is_valid, errors = validate_tp_sl_prices(
                operation=operation,
                entry_price=exec_price,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
            )
            if not is_valid:
                raise ValueError(
                    f"Invalid TP/SL in strategy: {'; '.join(errors)}. "
                    f"Please fix your strategy code."
                )

        # Open position (no longer pass TP/SL to position itself)
        account.open_position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=exec_price,
            leverage=leverage,
            timestamp=timestamp,
            fee=fee,
        )

        # Create independent TP/SL orders for this position
        close_side = "sell" if side == "long" else "buy"
        if tp_price:
            account.add_pending_order(
                symbol=symbol,
                side=close_side,
                order_type="take_profit",
                trigger_price=tp_price,
                size=size,
                entry_price=exec_price,
                timestamp=timestamp,
            )
        if sl_price:
            account.add_pending_order(
                symbol=symbol,
                side=close_side,
                order_type="stop_loss",
                trigger_price=sl_price,
                size=size,
                entry_price=exec_price,
                timestamp=timestamp,
            )

        return BacktestTradeRecord(
            timestamp=timestamp,
            trigger_type=trigger_type,
            symbol=symbol,
            operation=operation,
            side=side,
            entry_price=exec_price,
            size=size,
            leverage=leverage,
            fee=fee,
            reason=getattr(decision, 'reason', ''),
            pool_name=pool_name,
            triggered_signals=triggered_signals or [],
        )

    def _execute_close(
        self,
        account: VirtualAccount,
        symbol: str,
        current_price: float,
        timestamp: int,
        trigger_type: str,
        pool_name: Optional[str],
        triggered_signals: Optional[List[str]],
        reason: str = "",
        target_portion: float = 1.0,
    ) -> Optional[BacktestTradeRecord]:
        """
        Execute position close (full or partial).

        Args:
            target_portion: Portion of position to close (0.0-1.0).
                           1.0 = full close, 0.3 = close 30% of position.
        """
        pos = account.get_position(symbol)
        if not pos:
            return None

        # Calculate close size based on target_portion (matches real trading system)
        close_size = pos.size * target_portion
        if close_size <= 0:
            return None

        # Calculate execution price with slippage
        close_side = "sell" if pos.side == "long" else "buy"
        exec_price, _ = self.calculate_execution_price(current_price, close_side)

        # Calculate fee for the portion being closed
        notional = close_size * exec_price
        fee = self.calculate_fee(notional)

        # Store position info before closing
        entry_price = pos.entry_price
        leverage = pos.leverage
        side = pos.side
        entry_ts = pos.entry_timestamp

        # Close position (full or partial)
        if target_portion >= 1.0:
            # Full close
            pnl = account.close_position(symbol, exec_price, fee)
        else:
            # Partial close
            pnl = account.partial_close_position(
                symbol=symbol,
                size=close_size,
                exit_price=exec_price,
                fee=fee,
                entry_price=entry_price,
            )

        if pnl is None:
            return None

        # Calculate PnL percent
        entry_notional = close_size * entry_price
        pnl_percent = (pnl / entry_notional * 100) if entry_notional > 0 else 0

        return BacktestTradeRecord(
            timestamp=entry_ts,
            trigger_type=trigger_type,
            symbol=symbol,
            operation="close",
            side=side,
            entry_price=entry_price,
            size=close_size,
            leverage=leverage,
            exit_price=exec_price,
            exit_timestamp=timestamp,
            exit_reason="decision",
            pnl=pnl,
            pnl_percent=pnl_percent,
            fee=fee,
            reason=reason,
            pool_name=pool_name,
            triggered_signals=triggered_signals or [],
        )


"""
Program Backtest Engine

Event-driven backtest engine for Program Trader strategies.
Orchestrates trigger generation, strategy execution, and result calculation.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session

from .models import BacktestConfig, TriggerEvent, BacktestResult, BacktestTradeRecord, TriggerExecutionResult
from .virtual_account import VirtualAccount
from .execution_simulator import ExecutionSimulator
from .historical_data_provider import HistoricalDataProvider

logger = logging.getLogger(__name__)

# Interval to milliseconds mapping
INTERVAL_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


class ProgramBacktestEngine:
    """
    Event-driven backtest engine for Program Trader.

    Flow:
    1. Generate trigger events (signal + scheduled)
    2. Sort events by timestamp
    3. For each event:
       - Set historical data provider time
       - Check TP/SL triggers
       - Build MarketData
       - Execute strategy code
       - Simulate order execution
       - Update virtual account
    4. Calculate statistics
    """

    def __init__(self, db: Session):
        self.db = db

    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Run backtest with given configuration.

        Args:
            config: Backtest configuration

        Returns:
            BacktestResult with statistics and trade history
        """
        start_time = time.time()

        try:
            # 1. Generate signal trigger events (scheduled triggers are dynamic)
            signal_triggers = self._generate_trigger_events(config)

            # Allow backtest even with no signal triggers if scheduled triggers are enabled
            if not signal_triggers and not config.scheduled_interval_sec:
                return BacktestResult(
                    success=False,
                    error="No trigger events generated. Check signal pools and time range."
                )

            # 2. Initialize components
            account = VirtualAccount(initial_balance=config.initial_balance)
            simulator = ExecutionSimulator(
                slippage_percent=config.slippage_percent,
                fee_rate=config.fee_rate,
            )
            data_provider = HistoricalDataProvider(
                db=self.db,
                symbols=config.symbols,
                start_time_ms=config.start_time_ms,
                end_time_ms=config.end_time_ms,
                exchange=config.exchange,
            )

            # 3. Run event loop (returns all triggers including dynamic scheduled ones)
            trades, equity_curve, all_triggers = self._run_event_loop(
                config, signal_triggers, account, simulator, data_provider
            )

            # 4. Calculate statistics
            result = self._calculate_result(
                trades=trades,
                equity_curve=equity_curve,
                triggers=all_triggers,
                account=account,
                config=config,
            )
            result.execution_time_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return BacktestResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _generate_trigger_events(self, config: BacktestConfig) -> List[TriggerEvent]:
        """Generate signal trigger events only. Scheduled triggers are handled dynamically."""
        from services.signal_backtest_service import signal_backtest_service
        from services.market_regime_service import get_market_regime

        events = []

        # Generate signal triggers
        for pool_id in config.signal_pool_ids:
            for symbol in config.symbols:
                try:
                    pool_result = signal_backtest_service.backtest_pool(
                        self.db, pool_id, symbol,
                        config.start_time_ms, config.end_time_ms
                    )

                    if "error" in pool_result:
                        logger.warning(f"Signal backtest error for pool {pool_id}: {pool_result['error']}")
                        continue

                    for t in pool_result.get("triggers", []):
                        # Get market regime at trigger time
                        regime_data = None
                        try:
                            regime_result = get_market_regime(
                                self.db, symbol, "5m",
                                use_realtime=True,
                                timestamp_ms=t["timestamp"],
                                exchange=config.exchange
                            )
                            if regime_result:
                                regime_data = {
                                    "regime": regime_result.get("regime", "noise"),
                                    "conf": regime_result.get("confidence", 0.0),
                                    "direction": regime_result.get("direction", "neutral"),
                                    "reason": regime_result.get("reason", ""),
                                    "indicators": regime_result.get("indicators", {}),
                                }
                        except Exception as e:
                            logger.debug(f"Failed to get regime at {t['timestamp']}: {e}")

                        events.append(TriggerEvent(
                            timestamp=t["timestamp"],
                            trigger_type="signal",
                            symbol=symbol,
                            pool_id=pool_id,
                            pool_name=pool_result.get("pool_name"),
                            pool_logic=pool_result.get("logic"),
                            triggered_signals=t.get("triggered_signals", []),
                            market_regime=regime_data,
                        ))
                except Exception as e:
                    logger.error(f"Failed to get signal triggers for pool {pool_id}: {e}")

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Log signal triggers count (scheduled triggers are dynamic)
        logger.info(f"Generated {len(events)} signal trigger events")

        return events

    def _run_event_loop(
        self,
        config: BacktestConfig,
        signal_triggers: List[TriggerEvent],
        account: VirtualAccount,
        simulator: ExecutionSimulator,
        data_provider: HistoricalDataProvider,
    ) -> tuple:
        """
        Run the main event loop with dynamic scheduled trigger generation.

        Signal triggers have higher priority. Each trigger (signal or scheduled)
        resets the scheduled trigger timer, matching real-time execution behavior.
        """
        from program_trader.executor import SandboxExecutor
        from concurrent.futures import ThreadPoolExecutor

        backtest_thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest")
        executor = SandboxExecutor(timeout_seconds=5, thread_pool=backtest_thread_pool)
        trades: List[BacktestTradeRecord] = []
        equity_curve: List[Dict[str, Any]] = []
        all_triggers: List[TriggerEvent] = []  # Track all triggers for logging

        # Scheduled trigger state - now uses seconds directly
        scheduled_interval_ms = None
        if config.scheduled_interval_sec:
            scheduled_interval_ms = config.scheduled_interval_sec * 1000

        # Initialize: next scheduled trigger is start_time + interval
        last_trigger_time = config.start_time_ms

        def execute_trigger(trigger: TriggerEvent) -> int:
            """Execute a single trigger and return the trigger timestamp."""
            nonlocal last_trigger_time

            all_triggers.append(trigger)

            # Set current time
            data_provider.set_current_time(trigger.timestamp)

            # Get current prices
            prices = data_provider.get_current_prices(config.symbols)
            if not prices:
                return trigger.timestamp

            # Check TP/SL triggers first
            tp_sl_trades = simulator.check_tp_sl_triggers(account, prices, trigger.timestamp)
            trades.extend(tp_sl_trades)

            # Update equity after TP/SL
            account.update_equity(prices)
            # PRP-P1-1: funding shadow — 每trigger按持倉時間估算 (long付0.01%/8h, short收)
            # 只有在 account.funding_enabled=True 時才累加（預設 False，避免回測被保守估值污染）
            if getattr(account, 'funding_enabled', False):
                try:
                    _dt_h = max(0.0, (trigger.timestamp - last_trigger_time) / 3600000.0)
                    if _dt_h > 0 and _dt_h < 24:
                        for _sym, _pos in list(account.positions.items()):
                            _px = prices.get(_sym, 0) or 0
                            if _px <= 0:
                                continue
                            _notional = abs(_pos.size) * _px
                            _rate_h = 0.0001 / 8  # 保守 0.01%/8h
                            _pay = _notional * _rate_h * _dt_h
                            if _pos.side == "long":
                                account.total_funding_shadow += _pay
                            else:
                                account.total_funding_shadow -= _pay
                except Exception:
                    pass

            # 強平檢查：任何 trigger 都先確認倉位沒有達到維持保證金線。
            # 無此檢查時逆勢槓桿單無限扛單（實測 ZEC 1d 862d equity -1242%）。
            self._check_liquidations(account, prices, trigger.timestamp, trades, simulator)
            # Determine trigger symbol
            trigger_symbol = trigger.symbol if trigger.symbol else config.symbols[0]

            # Build MarketData for strategy (pass trades for recent_trades)
            market_data = self._build_market_data(
                account, data_provider, trigger, trigger_symbol, trades
            )

            # Execute strategy
            result = executor.execute(config.code, market_data, {})

            if result.success and result.decision:
                decision = result.decision
                symbol = decision.symbol or trigger_symbol
                current_price = prices.get(symbol, 0)

                if current_price > 0:
                    # Get signal names for logging
                    signal_names = [
                        s.get("signal_name", "") for s in (trigger.triggered_signals or [])
                    ]

                    trade = simulator.execute_decision(
                        decision=decision,
                        account=account,
                        current_price=current_price,
                        timestamp=trigger.timestamp,
                        trigger_type=trigger.trigger_type,
                        pool_name=trigger.pool_name,
                        triggered_signals=signal_names,
                    )
                    # collect reverse close stashed by simulator
                    stashed = getattr(simulator, "_last_close_trade", None)
                    if stashed is not None:
                        try:
                            delattr(simulator, "_last_close_trade")
                        except Exception:
                            try:
                                simulator._last_close_trade = None
                            except Exception:
                                pass
                        # avoid double count if already attached
                        if trade is not None and getattr(trade, "_close_trade", None) is stashed:
                            try:
                                delattr(trade, "_close_trade")
                            except Exception:
                                pass
                        trades.append(stashed)
                    # also handle attached close on trade
                    attached = getattr(trade, "_close_trade", None) if trade is not None else None
                    if attached is not None:
                        trades.append(attached)
                        try:
                            delattr(trade, "_close_trade")
                        except Exception:
                            pass
                    if trade:
                        trades.append(trade)

            # Update equity and record
            account.update_equity(prices)
            equity_curve.append({
                "timestamp": trigger.timestamp,
                "equity": account.equity,
                "balance": account.balance,
                "drawdown": account.max_drawdown,
            })

            return trigger.timestamp

        # Process signal triggers with dynamic scheduled triggers
        for signal_trigger in signal_triggers:
            # Before processing this signal, check if scheduled triggers should fire
            if scheduled_interval_ms:
                next_scheduled_time = last_trigger_time + scheduled_interval_ms
                while next_scheduled_time < signal_trigger.timestamp:
                    # Fire scheduled trigger
                    scheduled_trigger = TriggerEvent(
                        timestamp=next_scheduled_time,
                        trigger_type="scheduled",
                        symbol="",
                    )
                    last_trigger_time = execute_trigger(scheduled_trigger)
                    next_scheduled_time = last_trigger_time + scheduled_interval_ms

            # Process signal trigger (resets scheduled timer)
            last_trigger_time = execute_trigger(signal_trigger)

        # After all signal triggers, continue with remaining scheduled triggers until end_time
        if scheduled_interval_ms:
            next_scheduled_time = last_trigger_time + scheduled_interval_ms
            while next_scheduled_time <= config.end_time_ms:
                scheduled_trigger = TriggerEvent(
                    timestamp=next_scheduled_time,
                    trigger_type="scheduled",
                    symbol="",
                )
                last_trigger_time = execute_trigger(scheduled_trigger)
                next_scheduled_time = last_trigger_time + scheduled_interval_ms

        # Log final trigger counts
        signal_count = sum(1 for t in all_triggers if t.trigger_type == "signal")
        scheduled_count = sum(1 for t in all_triggers if t.trigger_type == "scheduled")
        logger.info(f"Executed {len(all_triggers)} triggers "
                   f"({signal_count} signal, {scheduled_count} scheduled)")

        backtest_thread_pool.shutdown(wait=False)
        return trades, equity_curve, all_triggers

    def estimate_total_triggers(
        self,
        config: BacktestConfig,
        signal_triggers: List[TriggerEvent],
    ) -> int:
        """
        Estimate total trigger count including dynamic scheduled triggers.
        Uses same algorithm as run_event_loop_generator but only counts.
        """
        if not config.scheduled_interval_sec:
            return len(signal_triggers)

        scheduled_interval_ms = config.scheduled_interval_sec * 1000
        total = 0
        last_trigger_time = config.start_time_ms

        for signal_trigger in signal_triggers:
            # Count scheduled triggers before this signal
            next_scheduled_time = last_trigger_time + scheduled_interval_ms
            while next_scheduled_time < signal_trigger.timestamp:
                total += 1
                last_trigger_time = next_scheduled_time
                next_scheduled_time = last_trigger_time + scheduled_interval_ms
            # Count signal trigger
            total += 1
            last_trigger_time = signal_trigger.timestamp

        # Count remaining scheduled triggers
        next_scheduled_time = last_trigger_time + scheduled_interval_ms
        while next_scheduled_time <= config.end_time_ms:
            total += 1
            last_trigger_time = next_scheduled_time
            next_scheduled_time = last_trigger_time + scheduled_interval_ms

        return total

    def run_event_loop_generator(
        self,
        config: BacktestConfig,
        signal_triggers: List[TriggerEvent],
        account: VirtualAccount,
        simulator: ExecutionSimulator,
        data_provider: HistoricalDataProvider,
    ):
        """
        Generator version of event loop for streaming progress.

        Yields TriggerExecutionResult for each trigger (signal or scheduled).
        Handles dynamic scheduled trigger generation with timer reset.
        """
        from program_trader.executor import SandboxExecutor
        from concurrent.futures import ThreadPoolExecutor

        backtest_thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest")
        executor = SandboxExecutor(timeout_seconds=5, thread_pool=backtest_thread_pool)

        try:
            yield from self._run_generator_body(
                config, signal_triggers, account, simulator, data_provider, executor
            )
        finally:
            backtest_thread_pool.shutdown(wait=False)

    def _run_generator_body(
        self,
        config: BacktestConfig,
        signal_triggers: List[TriggerEvent],
        account: VirtualAccount,
        simulator: ExecutionSimulator,
        data_provider: HistoricalDataProvider,
        executor,
    ):
        """Inner generator body for run_event_loop_generator."""
        all_trades: List[BacktestTradeRecord] = []

        # Scheduled trigger state - now uses seconds directly
        scheduled_interval_ms = None
        if config.scheduled_interval_sec:
            scheduled_interval_ms = config.scheduled_interval_sec * 1000

        last_trigger_time = config.start_time_ms

        def check_tp_sl_between_triggers(
            last_time_ms: int,
            current_time_ms: int,
        ) -> List[BacktestTradeRecord]:
            """Check TP/SL using 1m kline high/low for maximum accuracy."""
            all_tp_sl_trades = []

            # Get 1m klines between triggers for each symbol with positions
            for symbol in config.symbols:
                pos = account.get_position(symbol)
                if not pos:
                    continue

                # Use 1m klines for precise TP/SL detection
                klines = data_provider.get_klines_between(
                    symbol, last_time_ms, current_time_ms, "1m"
                )

                if klines:
                    trades = simulator.check_tp_sl_with_klines(
                        account, klines, pos.side, data_provider
                    )
                    all_tp_sl_trades.extend(trades)

            # Sort by exit timestamp
            all_tp_sl_trades.sort(key=lambda t: t.exit_timestamp or 0)
            return all_tp_sl_trades

        def execute_single_trigger(
            trigger: TriggerEvent,
            prev_trigger_time: int,
        ) -> TriggerExecutionResult:
            """Execute a single trigger and return result."""
            equity_before = account.equity

            # Set current time and clear query log
            data_provider.set_current_time(trigger.timestamp)
            data_provider.clear_query_log()

            # Get current prices
            prices = data_provider.get_current_prices(config.symbols)
            if not prices:
                return TriggerExecutionResult(
                    trigger=trigger,
                    trigger_symbol=config.symbols[0] if config.symbols else "",
                    prices={},
                    executor_result=None,
                    trade=None,
                    tp_sl_trades=[],
                    equity_before=equity_before,
                    equity_after=equity_before,
                    equity_after_tp_sl=equity_before,
                    unrealized_pnl=0,
                    data_queries=[],
                )

            # Check TP/SL using kline high/low between triggers (more accurate)
            tp_sl_trades = check_tp_sl_between_triggers(prev_trigger_time, trigger.timestamp)

            # Update equity after TP/SL and record it
            account.update_equity(prices)
            equity_after_tp_sl = account.equity

            # Determine trigger symbol
            trigger_symbol = trigger.symbol if trigger.symbol else config.symbols[0]

            # Snapshot account state BEFORE strategy execution (for logging)
            balance_before = account.balance
            positions_before = {
                k: {
                    "side": v.side,
                    "size": v.size,
                    "entry_price": v.entry_price,
                    "opened_at": v.entry_timestamp,
                }
                for k, v in account.positions.items()
            }
            used_margin_before = account.get_used_margin()
            margin_usage_percent_before = account.get_margin_usage_percent()

            # Build MarketData for strategy (pass all_trades for recent_trades)
            market_data = self._build_market_data(
                account, data_provider, trigger, trigger_symbol, all_trades
            )

            # Execute strategy
            result = executor.execute(config.code, market_data, {})

            trade = None
            if result.success and result.decision:
                decision = result.decision
                symbol = decision.symbol or trigger_symbol
                current_price = prices.get(symbol, 0)

                if current_price > 0 and decision.operation != "hold":
                    signal_names = [
                        s.get("signal_name", "") for s in (trigger.triggered_signals or [])
                    ]
                    trade = simulator.execute_decision(
                        decision=decision,
                        account=account,
                        current_price=current_price,
                        timestamp=trigger.timestamp,
                        trigger_type=trigger.trigger_type,
                        pool_name=trigger.pool_name,
                        triggered_signals=signal_names,
                    )
                    # collect reverse close stashed by simulator
                    stashed = getattr(simulator, "_last_close_trade", None)
                    if stashed is not None:
                        try:
                            delattr(simulator, "_last_close_trade")
                        except Exception:
                            try:
                                simulator._last_close_trade = None  # type: ignore
                            except Exception:
                                pass
                        if trade is not None and getattr(trade, "_close_trade", None) is stashed:
                            try:
                                delattr(trade, "_close_trade")
                            except Exception:
                                pass
                        all_trades.append(stashed)
                    attached = getattr(trade, "_close_trade", None) if trade is not None else None
                    if attached is not None:
                        all_trades.append(attached)
                        try:
                            delattr(trade, "_close_trade")
                        except Exception:
                            pass
                    if trade:
                        all_trades.append(trade)

            # Track TP/SL trades
            for tp_sl_trade in tp_sl_trades:
                all_trades.append(tp_sl_trade)

            # Update equity
            account.update_equity(prices)

            return TriggerExecutionResult(
                trigger=trigger,
                trigger_symbol=trigger_symbol,
                prices=prices,
                executor_result=result,
                trade=trade,
                tp_sl_trades=tp_sl_trades,
                equity_before=equity_before,
                equity_after=account.equity,
                equity_after_tp_sl=equity_after_tp_sl,
                unrealized_pnl=account.unrealized_pnl_total,
                balance_before=balance_before,
                positions_before=positions_before,
                used_margin_before=used_margin_before,
                margin_usage_percent_before=margin_usage_percent_before,
                data_queries=data_provider.get_query_log(),
            )

        # Process signal triggers with dynamic scheduled triggers
        for signal_trigger in signal_triggers:
            # Before processing this signal, fire any pending scheduled triggers
            if scheduled_interval_ms:
                next_scheduled_time = last_trigger_time + scheduled_interval_ms
                while next_scheduled_time < signal_trigger.timestamp:
                    scheduled_trigger = TriggerEvent(
                        timestamp=next_scheduled_time,
                        trigger_type="scheduled",
                        symbol="",
                    )
                    exec_result = execute_single_trigger(scheduled_trigger, last_trigger_time)
                    last_trigger_time = scheduled_trigger.timestamp
                    yield exec_result
                    next_scheduled_time = last_trigger_time + scheduled_interval_ms

            # Process signal trigger (resets scheduled timer)
            exec_result = execute_single_trigger(signal_trigger, last_trigger_time)
            last_trigger_time = signal_trigger.timestamp
            yield exec_result

        # After all signal triggers, continue with remaining scheduled triggers
        if scheduled_interval_ms:
            next_scheduled_time = last_trigger_time + scheduled_interval_ms
            while next_scheduled_time <= config.end_time_ms:
                scheduled_trigger = TriggerEvent(
                    timestamp=next_scheduled_time,
                    trigger_type="scheduled",
                    symbol="",
                )
                exec_result = execute_single_trigger(scheduled_trigger, last_trigger_time)
                last_trigger_time = scheduled_trigger.timestamp
                yield exec_result
                next_scheduled_time = last_trigger_time + scheduled_interval_ms

    def _check_liquidations(
        self,
        account,
        prices,
        timestamp: int,
        trades: list,
        simulator,
    ) -> None:
        """HL-style isolated-margin liquidation check (one-liner hook).

        無強平的回測是假的：逆勢槓桿單在真實市場早被砍掉，不能無限扛單。
        維持保證金率 50%（與 VirtualAccount.get_maintenance_margin 一致），
        權益 <= 維持保證金即市價全平，當根 trigger 收一筆強平費用。
        """
        for symbol in list(account.positions.keys()):
            px = prices.get(symbol, 0) or 0
            if px <= 0:
                continue
            pos = account.positions.get(symbol)
            if pos is None or pos.size <= 0:
                continue
            pos.update_pnl(px)
            eq = account.initial_balance + account.realized_pnl_total
            eq += sum(
                p.update_pnl(prices[s]) for s, p in account.positions.items()
                if s in prices and prices[s]
            )
            eq -= account.total_fees
            if eq <= pos.margin_used * 0.5:
                rec = simulator._execute_close(
                    account=account, symbol=symbol, current_price=px,
                    timestamp=timestamp, trigger_type="scheduled", pool_name="",
                    triggered_signals=[], reason="liquidation", target_portion=1.0,
                )
                if rec is not None:
                    rec.exit_reason = "liquidation"
                    trades.append(rec)

    def _build_market_data(
        self,
        account: VirtualAccount,
        data_provider: HistoricalDataProvider,
        trigger: TriggerEvent,
        trigger_symbol: str,
        recent_trades: List[BacktestTradeRecord] = None,
    ) -> Any:
        """Build MarketData object for strategy execution."""
        from program_trader.models import MarketData, Position, Trade
        from datetime import datetime, timezone

        # Convert virtual positions to Position objects
        positions = {}
        current_time_ms = trigger.timestamp  # Use trigger timestamp as current time in backtest
        for symbol, vpos in account.positions.items():
            # Calculate position timing information
            opened_at = vpos.entry_timestamp if vpos.entry_timestamp else None
            opened_at_str = None
            holding_duration_seconds = None
            holding_duration_str = None

            if opened_at and opened_at > 0:
                # Format opened_at as UTC string
                utc_dt = datetime.fromtimestamp(opened_at / 1000, tz=timezone.utc)
                opened_at_str = utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')

                # Calculate holding duration
                holding_duration_seconds = (current_time_ms - opened_at) / 1000
                hours = int(holding_duration_seconds // 3600)
                minutes = int((holding_duration_seconds % 3600) // 60)
                if hours > 0:
                    holding_duration_str = f"{hours}h {minutes}m"
                else:
                    holding_duration_str = f"{minutes}m"

            positions[symbol] = Position(
                symbol=symbol,
                side=vpos.side,
                size=vpos.size,
                entry_price=vpos.entry_price,
                unrealized_pnl=vpos.unrealized_pnl,
                leverage=vpos.leverage,
                liquidation_price=0,
                opened_at=opened_at,
                opened_at_str=opened_at_str,
                holding_duration_seconds=holding_duration_seconds,
                holding_duration_str=holding_duration_str,
            )

        # Convert recent trades to Trade objects (last 20 closed trades)
        trade_objects = []
        if recent_trades:
            closed_trades = [t for t in recent_trades if t.exit_price is not None][-20:]
            for t in closed_trades:
                trade_objects.append(Trade(
                    symbol=t.symbol,
                    side=t.side.capitalize(),  # "long" -> "Long"
                    size=t.size,
                    price=t.entry_price,
                    timestamp=t.timestamp,
                    pnl=t.pnl,
                ))

        # Build triggered signals info - pass through ALL fields from backtest service
        # Strategy code needs: metric, current_value, direction, ratio, threshold, etc.
        triggered_signals = []
        if trigger.triggered_signals:
            for sig in trigger.triggered_signals:
                # Pass through all signal data as-is
                signal_data = dict(sig)  # Copy all fields
                # Ensure current_value is set (backtest service uses 'value')
                if "current_value" not in signal_data and "value" in signal_data:
                    signal_data["current_value"] = signal_data["value"]
                triggered_signals.append(signal_data)

        # Build trigger_market_regime from trigger.market_regime
        trigger_market_regime = None
        if trigger.market_regime:
            from program_trader.models import RegimeInfo
            mr = trigger.market_regime
            trigger_market_regime = RegimeInfo(
                regime=mr.get("regime", "noise"),
                conf=mr.get("conf", 0.0),
                direction=mr.get("direction", "neutral"),
                reason=mr.get("reason", ""),
                indicators=mr.get("indicators", {}),
            )

        return MarketData(
            available_balance=account.balance,
            total_equity=account.equity,
            used_margin=account.get_used_margin(),
            margin_usage_percent=account.get_margin_usage_percent(),
            maintenance_margin=account.get_maintenance_margin(),
            trigger_symbol=trigger_symbol,
            trigger_type=trigger.trigger_type,
            positions=positions,
            recent_trades=trade_objects,
            # Trigger context (detailed)
            signal_pool_name=trigger.pool_name or "",
            pool_logic=trigger.pool_logic or "OR",
            triggered_signals=triggered_signals,
            trigger_market_regime=trigger_market_regime,
            _data_provider=data_provider,
        )


    def _force_close_all(
        self,
        account: VirtualAccount,
        config: BacktestConfig,
        trades: List[BacktestTradeRecord],
        equity_curve: List[Dict[str, Any]],
        triggers: List[TriggerEvent],
    ) -> None:
        """
        Force close all open positions at the end of backtest (方案乙).

        為什麼需要：舊的 _calculate_stats 只統計 exit_price is not None 的交易，
        導致回測結束時仍持有的部位，其浮盈浮虧完全不進入 PnL / trades / PF / 勝率，
        但 MaxDD 走 equity curve（盯市）。兩種會計基礎混用會讓 GA 系統性偏好
        「在視窗內把盈虧實現掉」的高換手 genome。

        這裡在視窗終點用最後一根已收盤 K 線的價格強制平倉，走與正常平倉
        相同的 fee/slippage 模型，於是五個 per-trade 指標的定義與 gate 門檻
        語意保持不變（見 ab_validators.annual_validator）。

        不變式：呼叫後 account.positions 必須為空，且
        account.equity - initial_balance == Σ(所有平倉交易的 pnl)。
        """
        if not account.positions:
            return

        end_time_ms = config.end_time_ms
        end_time_sec = end_time_ms // 1000

        # 價格一律取 <= end_time_ms 的最後一根「已收盤」K 線收盤價。
        # 不用 data_provider.get_current_prices()：那裡的 current time 是最後一次
        # trigger 的時間戳，會系統性地漏掉最後數小時的部位，也讓強平價取到
        # 未收盤的變動價格，造成候選間不可比。
        prices: Dict[str, float] = {}
        for symbol in account.positions:
            # 主週期取 config.kline_period（預設 1h 保持舊行為）。寫死 1h 會讓
            # 2h/4h/1d 回測在只有粗週期資料的區間（如 2026-04 之前的乾淨區）
            # 拿不到價格，部位不平倉 -> 會計恆等式破裂（實測 ZEC DD 216%）。
            row = self.db.execute(text(
                "SELECT close_price FROM crypto_klines "
                "WHERE symbol = :s AND exchange = :ex AND period = :p "
                "AND timestamp <= :t ORDER BY timestamp DESC LIMIT 1"
            ), {"s": symbol, "ex": config.exchange, "t": end_time_sec,
                "p": config.kline_period}).fetchone()
            if row and row[0] is not None:
                prices[symbol] = float(row[0])
            else:
                # fallback：允許任何已存在的週期，但必須 <= end_time_sec。
                # 不要只列細週期：乾淨區沒有 1h/5m/15m，只有 2h/4h/1d。
                row2 = self.db.execute(text(
                    "SELECT close_price, timestamp, period FROM crypto_klines "
                    "WHERE symbol = :s AND exchange = :ex "
                    "AND timestamp <= :t "
                    "ORDER BY timestamp DESC LIMIT 1"
                ), {"s": symbol, "ex": config.exchange, "t": end_time_sec}).fetchone()
                if row2 and row2[0] is not None:
                    prices[symbol] = float(row2[0])
                    if int(row2[1]) < end_time_sec - 7 * 86400:
                        logger.warning(
                            "force-close %s: 價格過舊 (%s 秒前)，結果不可比",
                            symbol, end_time_sec - int(row2[1]),
                        )
                else:
                    # 沒有任何可用價格 -> 明確記錄，不靜默跳過。
                    # 保留部位會讓會計恆等式失敗，比跳過更容易察覺。
                    logger.error(
                        "force-close: %s 在 end_time 之前沒有任何可用的收盤價，"
                        "該部位不會被平倉，會計恆等式將失敗", symbol,
                    )

        # 與正常平倉完全相同的 ExecutionSimulator 參數，確保 fee/slippage 一致。
        simulator = ExecutionSimulator(
            slippage_percent=config.slippage_percent,
            fee_rate=config.fee_rate,
        )

        forced = 0
        for symbol, price in prices.items():
            if symbol not in account.positions:
                continue
            if account.positions[symbol].size <= 0:
                continue
            record = simulator._execute_close(
                account=account,
                symbol=symbol,
                current_price=price,
                timestamp=end_time_ms,
                trigger_type="scheduled",
                pool_name="",
                triggered_signals=[],
                reason="end_of_backtest",
                target_portion=1.0,
            )
            if record is not None:
                record.exit_reason = "end_of_backtest"
                trades.append(record)
                forced += 1

        if forced:
            # 只在全部平倉後補一個終點，並且以最後一根的價格更新 equity，
            # 讓 equity_curve 的終點與平倉後的帳面一致。
            final_prices = dict(prices)
            account.update_equity(final_prices)
            equity_curve.append({
                "timestamp": end_time_ms,
                "equity": account.equity,
                "balance": account.balance,
                "drawdown": account.max_drawdown,
            })
            logger.info("force-close: %d 個未平倉部位已在終點平倉", forced)

    def _resample_equity_daily(
        self,
        equity_curve: List[Dict[str, Any]],
        start_time_ms: int,
        end_time_ms: int,
    ) -> List[float]:
        """
        Resample equity curve from irregular trigger timestamps to fixed daily 00:00 UTC grid.
        Uses linear interpolation between points.
        
        Returns list of equity values at daily 00:00 UTC, suitable for daily return calculation.
        """
        if not equity_curve or len(equity_curve) < 2:
            return []
        
        import math
        from datetime import datetime, timezone
        
        # Convert to (timestamp_ms, equity) pairs, sorted by timestamp
        points = []
        for p in equity_curve:
            ts = p.get("timestamp", 0)
            eq = p.get("equity", 0.0)
            if ts > 0 and eq > 0:
                points.append((ts, eq))
        
        if len(points) < 2:
            return []
        
        points.sort(key=lambda x: x[0])
        
        # Determine daily grid: start at 00:00 UTC of start day, end at 00:00 UTC of end day
        start_day = (start_time_ms // 86400000) * 86400000
        end_day = (end_time_ms // 86400000) * 86400000
        
        # Build daily equity values by linear interpolation
        daily_equity = []
        point_idx = 0
        
        for day_ts in range(start_day, end_day + 86400000, 86400000):
            # Find the two points surrounding this day_ts
            # Move point_idx forward until we find the right interval
            while point_idx + 1 < len(points) and points[point_idx + 1][0] <= day_ts:
                point_idx += 1
            
            if point_idx == 0:
                # Before first point - use first point's equity
                daily_equity.append(points[0][1])
            elif point_idx == len(points) - 1:
                # After last point - use last point's equity
                daily_equity.append(points[-1][1])
            else:
                # Interpolate between points[point_idx] and points[point_idx + 1]
                t1, e1 = points[point_idx]
                t2, e2 = points[point_idx + 1]
                if t2 == t1:
                    daily_equity.append(e1)
                else:
                    # Linear interpolation
                    alpha = (day_ts - t1) / (t2 - t1)
                    interp_eq = e1 + alpha * (e2 - e1)
                    daily_equity.append(interp_eq)
        
        return daily_equity
    def _calculate_result(
        self,
        trades: List[BacktestTradeRecord],
        equity_curve: List[Dict[str, Any]],
        triggers: List[TriggerEvent],
        account: VirtualAccount,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Calculate backtest statistics."""
        # Accounting: funding shadow -> deduct from balance/equity before metrics
        # This ensures funding cost is reflected in PnL, maxDD, Sharpe consistently.
        if hasattr(account, 'total_funding_shadow') and account.total_funding_shadow != 0:
            account.total_fees += account.total_funding_shadow
            account.total_funding_shadow = 0.0
            # Recalculate equity after funding deduction
            account.update_equity({})  # prices empty ok, just refreshes equity from realized/unrealized/fees
        
        # Force close all open positions at end of backtest
        self._force_close_all(account, config, trades, equity_curve, triggers)
        
        # Filter closed trades (with exit_price) - NOW INCLUDES FORCED CLOSES
        closed_trades = [t for t in trades if t.exit_price is not None]

        # Basic counts
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        # PnL calculations
        # NOTE: t.pnl 是「平倉前未扣本次出場費」的毛利（見 VirtualAccount.close_position
        # 回傳的是 fee 前的 realized_pnl，fee 另外記在 account.total_fees）。
        # 舊碼 total_pnl = sum(t.pnl) 是稅前毛值，但 MaxDD / equity 走的是
        # 扣費後的 account.equity -> 兩種會計基礎不一致，且 fee + funding 完全不
        # 反映在 PnL 上（實測 BTC 買入持有 10 天：毛 941.56、淨 901.82，差 39.74）。
        # 這裡改為淨值，使 PnL 與 equity/MaxDD/Sharpe 同一會計基礎。
        total_pnl_gross = sum(t.pnl for t in closed_trades)
        total_pnl = total_pnl_gross - account.total_fees
        total_pnl_percent = (total_pnl / config.initial_balance * 100) if config.initial_balance > 0 else 0

        # Win rate
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Net-basis PF: attribute account-level fees (entry + exit + funding shadow)
        # pro-rata across closed trades by notional. Per-trade fee records only
        # hold the exit leg (t.fee), so dividing total_fees by notional is the
        # only consistent basis. All-loss/all-win edge cases stay None-safe.
        total_notional = sum((t.size or 0) * (t.entry_price or 0) for t in closed_trades)
        fee_rate = (abs(account.total_fees) / total_notional) if total_notional > 0 else 0.0
        net_profits = [t.pnl - (t.size or 0) * (t.entry_price or 0) * fee_rate for t in winning_trades]
        net_losses = [t.pnl - (t.size or 0) * (t.entry_price or 0) * fee_rate for t in losing_trades]
        # Re-split: a fee-adjusted winner may flip to a loser and vice versa.
        net_wins = [p for p in net_profits + net_losses if p > 0]
        net_ls = [p for p in net_profits + net_losses if p <= 0]
        total_profit = sum(net_wins)
        total_loss = abs(sum(net_ls))
        # When total_loss is 0, profit_factor is undefined (use None to avoid JSON Infinity issue)
        profit_factor = (total_profit / total_loss) if total_loss > 0 else None

        # Average win/loss
        avg_win = (total_profit / len(net_wins)) if net_wins else 0
        avg_loss = (total_loss / len(net_ls)) if net_ls else 0

        # Largest win/loss
        largest_win = max(net_wins, default=0)
        largest_loss = min(net_ls, default=0)

        # Trigger counts
        signal_triggers = sum(1 for t in triggers if t.trigger_type == "signal")
        scheduled_triggers = sum(1 for t in triggers if t.trigger_type == "scheduled")

        # Sharpe ratio (annualized) - resample equity curve to daily grid first
        # Crypto trades 24/7, so use 365 days, not 252.
        # The equity_curve points are at trigger times (irregular), so resample to daily 00:00 UTC.
        sharpe_ratio = 0.0
        if equity_curve and len(equity_curve) > 1:
            daily_equity = self._resample_equity_daily(equity_curve, config.start_time_ms, config.end_time_ms)
            if len(daily_equity) > 1:
                returns = []
                for i in range(1, len(daily_equity)):
                    prev_eq = daily_equity[i-1]
                    curr_eq = daily_equity[i]
                    if prev_eq > 0:
                        returns.append((curr_eq - prev_eq) / prev_eq)
                if returns:
                    import statistics
                    mean_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                    if std_return > 0:
                        # Annualize: daily returns * sqrt(365)
                        sharpe_ratio = (mean_return / std_return) * (365 ** 0.5)

        # Accounting identity: final equity - initial_balance == sum of all closed trade PnL
        # This is the fundamental check that our force-close + metrics are consistent.
        all_closed = [t for t in trades if t.exit_price is not None]
        sum_closed_pnl = sum(t.pnl for t in all_closed)
        equity_diff = account.equity - config.initial_balance
        # equity = initial + Σ(平倉毛利) - total_fees（含 funding），因此恆等式要帶費用。
        expected = sum_closed_pnl - account.total_fees
        if abs(equity_diff - expected) > 1e-6:
            import logging
            logging.warning(
                "Accounting identity violation: equity_diff=%.6f expected=%.6f "
                "(gross=%.6f fees=%.6f) diff=%.6f. Bug in force-close or metrics.",
                equity_diff, expected, sum_closed_pnl, account.total_fees,
                equity_diff - expected
            )

        return BacktestResult(
            success=True,
            error=None,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            max_drawdown=account.max_drawdown,
            max_drawdown_percent=account.max_drawdown_percent * 100,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_triggers=len(triggers),
            signal_triggers=signal_triggers,
            scheduled_triggers=scheduled_triggers,
            equity_curve=equity_curve,
            trades=trades,
            trigger_log=triggers,
            start_time=config.start_time,
            end_time=config.end_time,
        )


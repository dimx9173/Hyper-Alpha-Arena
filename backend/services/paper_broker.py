#!/usr/bin/env python3
"""
Paper Trading Broker — 用 Hyperliquid mainnet 行情，資金與持倉完全本地計算。

設計目標
--------
解決「回測用 binance 資料、實盤在 testnet 執行」的落差問題。

實測（2026-10-03）Hyperliquid mainnet 與 Binance 價格一致度：
  BTC ETH ZEC SOL NEAR DOGE HYPE BNB SUI NIL ADA TAO WLD AVAX
  PENGU ARB TRUMP FIL AAVE PUMP —— 20/20 全部在 0.3% 以內。

而 testnet 的 ZEC 是 138.65，mainnet 是 1305.25（差 9.4 倍），
testnet 與 binance 的價格結構不一致，回測績效無法預測 testnet 結果。

因此：行情走 mainnet（與回測同源），帳務走本地（不碰鏈上、不花真錢）。

與真實 TradingClient 的差異（有意为之）
----------------------------------------
1. 不簽名、不廣播任何交易，只更新本地帳本
2. 滑價與手續費是模型參數，不是真實成交結果
3. TP/SL 在每次 check() 時用 mainnet 現價觸發判定，
   不像鏈上那樣由交易所托管觸發
4. 倉位記錄在 DB 的 paper_* 表，與鏈上狀態完全隔離

這是「計算」，不是「交易」。產出的績效數字是模型輸出，不是實盤結果。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# 成本模型
# =====================================================================

@dataclass
class FeeModel:
    """手續費與滑價模型。

    預設值取自 Hyperliquid mainnet 的實際費率量級：
    - maker 0.015% / taker 0.045%（一般用戶）
    - 滑價 2bp（假設流動性充足的小幣）
    """
    maker_fee_bps: float = 1.5      # 0.015%
    taker_fee_bps: float = 4.5      # 0.045%
    slippage_bps: float = 2.0       # 單邊 2bp

    def entry_price(self, mid: float, is_buy: bool) -> float:
        """進場成交價（含滑價）。買入滑高，賣出滑低。"""
        adj = self.slippage_bps / 10000.0
        return mid * (1 + adj) if is_buy else mid * (1 - adj)

    def exit_price(self, mid: float, is_buy: bool) -> float:
        """平倉成交價（含滑價）。賣出滑低，買回滑高。"""
        adj = self.slippage_bps / 10000.0
        return mid * (1 - adj) if is_buy else mid * (1 + adj)

    def fee(self, notional: float, is_maker: bool = False) -> float:
        bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        return abs(notional) * bps / 10000.0


# =====================================================================
# 持倉
# =====================================================================

@dataclass
class PaperPosition:
    symbol: str
    size: float                 # 正數 = 多，負數 = 空
    entry_price: float
    entry_time: str
    leverage: int = 1
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    margin: float = 0.0         # 佔用的保證金
    fees_paid: float = 0.0
    realized_pnl: float = 0.0   # 已實現損益（含出場時的平倉費用）
    tp_hit: bool = False
    sl_hit: bool = False

    @property
    def notional(self) -> float:
        return abs(self.size) * self.entry_price

    def unrealized_pnl(self, mark: float) -> float:
        direction = 1.0 if self.size >= 0 else -1.0
        return (mark - self.entry_price) * abs(self.size) * direction

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["notional"] = self.notional
        return d


@dataclass
class PaperTrade:
    """一筆完整的平倉紀錄。"""
    symbol: str
    side: str              # "long" | "short"
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    leverage: int
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str       # "take_profit" | "stop_loss" | "manual"
    bars_held: int = 0


# =====================================================================
# Broker
# =====================================================================

class PaperBroker:
    """模擬帳戶。所有狀態在記憶體 / DB，由呼叫端決定是否持久化。"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_model: Optional[FeeModel] = None,
        max_positions: int = 3,
        cooldown_hours: float = 4.0,
        max_total_leverage: float = 2.0,
        max_single_coin: float = 0.4,
        max_cluster: float = 0.6,
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.fee_model = fee_model or FeeModel()
        self.max_positions = max_positions
        self.cooldown_hours = float(cooldown_hours)
        self._max_total_leverage = float(max_total_leverage)
        self._max_single_coin = float(max_single_coin)
        self._max_cluster = float(max_cluster)

        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []

        # 風控護欄（與 program_execution_service 的限制對齊）
        self.CLUSTER = {"BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR"}

        # 同向加倉冷卻
        self._last_add_time: Dict[str, datetime] = {}

        self._record_equity(datetime.now(timezone.utc))

    # ---------------- 帳務查詢 ----------------

    def total_equity(self, marks: Optional[Dict[str, float]] = None) -> float:
        """總權益 = 錢包餘額 + 未實現損益。

        會計模型（與永續合約一致）：
          - self.cash 是「錢包餘額」（wallet balance）= 初始資金 + 已實現損益 − 費用
          - 開倉時 **不** 從 cash 扣保證金。保證金是交易所要求的抵押品，
            不是從帳戶轉出的資金；扣掉會讓權益低估一整個 margin。
          - margin 單獨記在 position.margin，用於計算可用餘額。

        錯誤做法（已修正）：open 時 cash -= margin，close 時 cash += margin。
        那會讓權益在每次開倉時憑空減少 margin，平倉時憑空加回，造成
        equity identity 崩潰。
        """
        unreal = 0.0
        for sym, pos in self.positions.items():
            mark = (marks or {}).get(sym, pos.entry_price)
            unreal += pos.unrealized_pnl(mark)
        return self.cash + unreal

    def used_margin(self) -> float:
        """已被持倉佔用的保證金總額。"""
        return sum(p.margin for p in self.positions.values())

    def available_balance(self, marks: Optional[Dict[str, float]] = None) -> float:
        """可用餘額 = 錢包餘額 − 已佔用保證金。"""
        return max(0.0, self.cash - self.used_margin())

    def get_account_state(self, marks: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        eq = self.total_equity(marks)
        avail = self.available_balance(marks)
        used = sum(p.margin for p in self.positions.values())
        return {
            "total_equity": round(eq, 4),
            "available_balance": round(avail, 4),
            "used_margin": round(used, 4),
            "maintenance_margin": 0.0,
            "margin_usage_percent": round(used / eq * 100, 2) if eq > 0 else 0.0,
            "withdrawal_available": round(avail, 4),
            "account_mode": "paper",
            "source": "paper_broker",
            "positions_count": len(self.positions),
        }

    def get_positions(self, marks: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        out = []
        for sym, p in self.positions.items():
            mark = (marks or {}).get(sym, p.entry_price)
            d = p.to_dict()
            d["mark_price"] = round(mark, 8)
            d["unrealized_pnl"] = round(p.unrealized_pnl(mark), 6)
            d["roe_pct"] = round(p.unrealized_pnl(mark) / p.margin * 100, 4) if p.margin else 0.0
            out.append(d)
        return out

    # ---------------- 風控 ----------------

    def _check_guards(self, symbol: str, size: float, price: float, leverage: int) -> Optional[str]:
        """回傳阻擋原因；None 表示通過。"""
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            return f"max_positions ({self.max_positions})"

        equity = self.total_equity()
        if equity <= 0:
            return "no_equity"

        new_notional = size * price

        # 單幣護欄
        existing = self.positions.get(symbol)
        add_notional = existing.notional if existing else 0.0
        if new_notional + add_notional > equity * self._max_single_coin:
            return (f"single_coin_cap: {new_notional + add_notional:.2f} > "
                    f"{equity * self._max_single_coin:.2f}")

        # 總暴露護欄
        total = sum(p.notional for p in self.positions.values())
        if total + new_notional > equity * self._max_total_leverage:
            return (f"exposure_cap: {total + new_notional:.2f} > "
                    f"{equity * self._max_total_leverage:.2f}")

        # 同簇護欄
        if symbol in self.CLUSTER:
            cluster = sum(p.notional for s, p in self.positions.items() if s in self.CLUSTER)
            if cluster + new_notional > equity * self._max_cluster:
                return (f"cluster_cap: {cluster + new_notional:.2f} > "
                        f"{equity * self._max_cluster:.2f}")

        # 保證金檢查
        margin_needed = new_notional / max(leverage, 1)
        if margin_needed > self.available_balance():
            return (f"insufficient_margin: need {margin_needed:.2f}, "
                    f"have {self.available_balance():.2f}")

        # 同向加倉冷卻
        now = datetime.now(timezone.utc)
        if existing is not None:
            last = self._last_add_time.get(symbol)
            if last is not None:
                elapsed = (now - last).total_seconds() / 3600.0
                if elapsed < self.cooldown_hours:
                    return f"cooldown: {self.cooldown_hours - elapsed:.1f}h remaining"

        return None

    # ---------------- 下單 ----------------

    def open_long(
        self,
        symbol: str,
        portion: float,
        mid_price: float,
        leverage: int = 1,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """開多倉。portion = 佔 available_balance 的比例。"""
        # portion 的分母是「開倉當下的可用餘額」，與真實執行路徑一致
        # （program_execution_service 也是用 available_balance × portion）。
        # 先做護欄檢查再算 size，否則護欄會用錯誤的 notional 判斷。
        fill = self.fee_model.entry_price(mid_price, is_buy=True)
        if fill <= 0:
            return {"success": False, "skipped": "bad_price", "symbol": symbol}

        avail = self.available_balance()
        margin = avail * portion
        size = (margin * leverage) / fill

        reason = self._check_guards(symbol, size, fill, leverage)
        if reason:
            return {"success": False, "skipped": reason, "symbol": symbol}

        fee = self.fee_model.fee(size * fill, is_maker=False)
        if margin + fee > avail:
            return {"success": False, "skipped": "insufficient_cash", "symbol": symbol}

        now = datetime.now(timezone.utc)
        existing = self.positions.get(symbol)
        if existing:
            # 加倉：加權平均成本
            total_size = existing.size + size
            existing.entry_price = (
                (existing.entry_price * existing.size + fill * size) / total_size
            )
            existing.size = total_size
            existing.margin += margin
            existing.fees_paid += fee
        else:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                size=size,
                entry_price=fill,
                entry_time=now.isoformat(),
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
                margin=margin,
                fees_paid=fee,
            )
        # 只扣手續費。保證金不是轉出資金，見 total_equity 的說明。
        self.cash -= fee
        self._last_add_time[symbol] = now
        self._record_equity(now)

        return {
            "success": True,
            "symbol": symbol,
            "side": "long",
            "size": size,
            "entry_price": round(fill, 8),
            "margin": round(margin, 4),
            "fee": round(fee, 6),
            "leverage": leverage,
        }

    def close_position(
        self,
        symbol: str,
        mid_price: float,
        reason: str = "manual",
    ) -> Dict[str, Any]:
        """平掉一個持倉。"""
        pos = self.positions.get(symbol)
        if pos is None:
            return {"success": False, "error": f"no position in {symbol}"}

        is_long = pos.size >= 0
        fill = self.fee_model.exit_price(mid_price, is_buy=not is_long)
        fee = self.fee_model.fee(pos.size * fill, is_maker=False)

        gross = (fill - pos.entry_price) * pos.size * (1.0 if is_long else -1.0)
        net = gross - fee - pos.fees_paid

        # 已實現損益入帳。保證金釋放不影響 cash（它從未被扣除）。
        self.cash += gross - fee
        pos.realized_pnl += (gross - fee)

        now = datetime.now(timezone.utc)
        entry_dt = datetime.fromisoformat(pos.entry_time)
        bars = int((now - entry_dt).total_seconds() // 3600)

        self.trades.append(PaperTrade(
            symbol=symbol,
            side="long" if is_long else "short",
            entry_time=pos.entry_time,
            exit_time=now.isoformat(),
            entry_price=round(pos.entry_price, 8),
            exit_price=round(fill, 8),
            size=pos.size,
            leverage=pos.leverage,
            gross_pnl=round(gross, 6),
            fees=round(fee + pos.fees_paid, 6),
            net_pnl=round(net, 6),
            exit_reason=reason,
            bars_held=bars,
        ))

        del self.positions[symbol]
        self._last_add_time.pop(symbol, None)
        self._record_equity(now)

        return {
            "success": True,
            "symbol": symbol,
            "exit_price": round(fill, 8),
            "gross_pnl": round(gross, 6),
            "fees": round(fee, 6),
            "net_pnl": round(net, 6),
            "reason": reason,
        }

    # ---------------- 週期性檢查 ----------------

    def check_exits(self, marks: Dict[str, float]) -> List[Dict[str, Any]]:
        """用最新價檢查 TP/SL。這是 paper 模式的核心：模擬交易所的託管觸發。"""
        results = []
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if pos is None:
                continue
            mark = marks.get(symbol)
            if mark is None or mark <= 0:
                continue

            exit_price = None
            reason = None

            if pos.take_profit and pos.size >= 0 and mark >= pos.take_profit:
                exit_price, reason = pos.take_profit, "take_profit"
                pos.tp_hit = True
            elif pos.stop_loss and pos.size >= 0 and mark <= pos.stop_loss:
                exit_price, reason = pos.stop_loss, "stop_loss"
                pos.sl_hit = True
            elif pos.take_profit and pos.size < 0 and mark <= pos.take_profit:
                exit_price, reason = pos.take_profit, "take_profit"
            elif pos.stop_loss and pos.size < 0 and mark >= pos.stop_loss:
                exit_price, reason = pos.stop_loss, "stop_loss"
                pos.sl_hit = True

            if exit_price is not None:
                # 用觸發價而非市價，模擬 TP/SL 的限價成交
                results.append(self._close_at(symbol, exit_price, reason))
            else:
                pos.mark_price = mark

        if results:
            self._record_equity(datetime.now(timezone.utc))
        return results

    def _close_at(self, symbol: str, price: float, reason: str) -> Dict[str, Any]:
        """以指定價格平倉（用於 TP/SL 觸發）。"""
        pos = self.positions[symbol]
        is_long = pos.size >= 0
        fee = self.fee_model.fee(pos.size * price, is_maker=True)  # TP/SL 視為 maker
        gross = (price - pos.entry_price) * pos.size * (1.0 if is_long else -1.0)
        net = gross - fee - pos.fees_paid

        self.cash += gross - fee
        now = datetime.now(timezone.utc)
        entry_dt = datetime.fromisoformat(pos.entry_time)
        bars = int((now - entry_dt).total_seconds() // 3600)

        self.trades.append(PaperTrade(
            symbol=symbol, side="long" if is_long else "short",
            entry_time=pos.entry_time, exit_time=now.isoformat(),
            entry_price=round(pos.entry_price, 8), exit_price=round(price, 8),
            size=pos.size, leverage=pos.leverage,
            gross_pnl=round(gross, 6), fees=round(fee + pos.fees_paid, 6),
            net_pnl=round(net, 6), exit_reason=reason, bars_held=bars,
        ))
        del self.positions[symbol]
        self._last_add_time.pop(symbol, None)
        return {"success": True, "symbol": symbol, "net_pnl": round(net, 6),
                "reason": reason, "exit_price": round(price, 8)}

    # ---------------- 績效統計 ----------------

    def performance(self, marks: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """計算績效摘要。欄位對齊 backtest gate 的需求。"""
        eq = self.total_equity(marks)
        ret_pct = (eq / self.initial_capital - 1) * 100 if self.initial_capital else 0.0

        closed = self.trades
        wins = [t for t in closed if t.net_pnl > 0]
        losses = [t for t in closed if t.net_pnl <= 0]

        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_win > 0 else 0.0)

        # 最大回撤
        peak, max_dd = self.initial_capital, 0.0
        for pt in self.equity_curve:
            peak = max(peak, pt["equity"])
            dd = (peak - pt["equity"]) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(eq, 4),
            "total_return_pct": round(ret_pct, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 4) if closed else 0.0,
            "profit_factor": round(pf, 4) if pf != float("inf") else None,
            "avg_win": round(gross_win / len(wins), 6) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 6) if losses else 0.0,
            "total_fees": round(sum(t.fees for t in closed), 6),
            "open_positions": len(self.positions),
        }

    def _record_equity(self, ts: datetime) -> None:
        eq = self.total_equity()
        self.equity_curve.append({
            "t": ts.isoformat(),
            "equity": round(eq, 4),
            "cash": round(self.cash, 4),
            "positions": len(self.positions),
        })

    # ---------------- 持久化 ----------------

    def to_state(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "fee_model": asdict(self.fee_model),
            "max_positions": self.max_positions,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
            "equity_curve": self.equity_curve,
            "last_add_time": {k: v.isoformat() for k, v in self._last_add_time.items()},
        }

    def save_to(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_state(), f, indent=2, default=str)
        logger.info("paper state saved: %s (equity=%.2f, trades=%d)",
                    path, self.total_equity(), len(self.trades))

    @classmethod
    def load_from(cls, path: str) -> "PaperBroker":
        with open(path) as f:
            st = json.load(f)
        b = cls(initial_capital=st["initial_capital"],
                fee_model=FeeModel(**st["fee_model"]),
                max_positions=st.get("max_positions", 3))
        b.cash = st["cash"]
        for sym, pd in st.get("positions", {}).items():
            known = {f for f in PaperPosition.__dataclass_fields__ if f != "mark_price"}
            b.positions[sym] = PaperPosition(**{k: v for k, v in pd.items() if k in known})
        for td in st.get("trades", []):
            known = set(PaperTrade.__dataclass_fields__)
            b.trades.append(PaperTrade(**{k: v for k, v in td.items() if k in known}))
        b.equity_curve = st.get("equity_curve", [])
        for sym, ts in st.get("last_add_time", {}).items():
            b._last_add_time[sym] = datetime.fromisoformat(ts)
        return b

#!/usr/bin/env python3
"""
PaperTradingClient — 讓 PaperBroker 對執行服務來說「看起來像」真實交易 client。

存在的理由
----------
`program_execution_service` 透過 `get_hyperliquid_client()` 取得一個 client，
然後呼叫它的 `get_account_state()` / `get_positions()` / `place_order()` /
`place_tp_sl()` 等方法。若要讓 paper 模式走同一條執行路徑（而不是在
service 裡加一堆 if/else 分岔），最乾淨的做法是提供一個介面相同的 adapter。

行情來自 Hyperliquid **mainnet**（與回測用的 Binance 價格一致，實測 20/20
標的都在 0.3% 內）。資金與持倉完全本地計算，不簽名、不廣播。

與真實 client 的差異（有意為之）
--------------------------------
1. place_order 永不廣播，只更新本地帳本
2. 滑價與手續費是模型參數
3. TP/SL 每次呼叫時用 mainnet 現價判定，不像鏈上由交易所托管
4. 狀態存於本地檔案，可透過 refresh() 重載
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.paper_broker import PaperBroker, FeeModel

logger = logging.getLogger(__name__)

# 狀態檔案位置：放在 /app/data 是因為那是 volume mount，重啟不會遺失
DEFAULT_STATE_PATH = "/app/data/paper_state.json"


class PaperTradingClient:
    """模擬 Hyperliquid trading client 的本地 paper broker。

    介面刻意對齊 services/hyperliquid_trading_client.py 的常用方法，
    讓 program_execution_service 不需要知道自己在 paper 模式。
    """

    def __init__(
        self,
        account_id: int,
        state_path: Optional[str] = None,
        initial_capital: float = 10000.0,
        environment: str = "mainnet",
    ):
        self.account_id = account_id
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.environment = environment
        self.initial_capital = initial_capital
        self._lock = threading.Lock()
        self._order_seq = 0

        self.broker = self._load_or_create()

    # ---------------- 狀態管理 ----------------

    def _load_or_create(self) -> PaperBroker:
        path = f"{self.state_path}.{self.account_id}"
        if os.path.exists(path):
            try:
                b = PaperBroker.load_from(path)
                logger.info("[PaperClient] loaded state for account %s: "
                            "equity=%.2f positions=%d trades=%d",
                            self.account_id, b.total_equity(),
                            len(b.positions), len(b.trades))
                return b
            except Exception as e:
                logger.warning("[PaperClient] failed to load %s (%s), starting fresh", path, e)
        b = PaperBroker(initial_capital=self.initial_capital, fee_model=FeeModel())
        logger.info("[PaperClient] new paper account %s: capital=%.2f",
                    self.account_id, self.initial_capital)
        return b

    def save(self) -> None:
        """把狀態寫到磁碟。每筆交易後呼叫。"""
        path = f"{self.state_path}.{self.account_id}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.broker.save_to(path)

    def reset(self) -> None:
        """清空帳戶，回到初始資金。"""
        self.broker = PaperBroker(initial_capital=self.initial_capital, fee_model=FeeModel())
        self.save()
        logger.warning("[PaperClient] account %s reset", self.account_id)

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"paper-{self.account_id}-{int(datetime.now(timezone.utc).timestamp())}-{self._order_seq}"

    # ---------------- 行情 ----------------

    def _mainnet_price(self, symbol: str) -> Optional[float]:
        try:
            import requests
            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "allMids"},
                timeout=15,
            )
            r.raise_for_status()
            mids = r.json()
            v = mids.get(symbol)
            return float(v) if v is not None else None
        except Exception as e:
            logger.warning("[PaperClient] price fetch failed for %s: %s", symbol, e)
            return None

    def _marks(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        syms = symbols or list(self.broker.positions.keys())
        out = {}
        for s in syms:
            px = self._mainnet_price(s)
            if px:
                out[s] = px
        return out

    # ---------------- 介面：帳戶 ----------------

    def get_account_state(self, db=None) -> Dict[str, Any]:
        marks = self._marks()
        st = self.broker.get_account_state(marks)
        st["environment"] = "paper"
        st["data_source"] = "hyperliquid_mainnet"
        st["wallet_address"] = f"paper-{self.account_id}"
        return st

    def get_positions(self, db=None, include_timing: bool = False) -> List[Dict[str, Any]]:
        marks = self._marks()
        out = []
        for p in self.broker.get_positions(marks):
            # 對齊真實 client 的回傳格式
            out.append({
                "symbol": p["symbol"],
                "size": p["size"],
                "direction": "long" if p["size"] >= 0 else "short",
                "entry_price": p["entry_price"],
                "mark_price": p["mark_price"],
                "unrealized_pnl": p["unrealized_pnl"],
                "roe_pct": p["roe_pct"],
                "margin": p["margin"],
                "leverage": p["leverage"],
                "take_profit": p["take_profit"],
                "stop_loss": p["stop_loss"],
            })
        return out

    def get_open_orders(self, db=None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Paper 模式下沒有掛單。TP/SL 由本地 check_exits 判定。"""
        return []

    # ---------------- 介面：下單 ----------------

    def place_order(
        self,
        db,
        symbol: str,
        side: str,
        size: Optional[float] = None,
        price: Optional[float] = None,
        portion_of_balance: Optional[float] = None,
        leverage: int = 1,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """模擬下單。回傳格式對齊真實 client。"""
        with self._lock:
            mid = price or self._mainnet_price(symbol)
            if mid is None:
                return {"success": False, "error": f"no price for {symbol}"}

            if side.lower() in ("sell", "short"):
                # 開空：paper broker 目前只支援多頭，先平倉語意處理
                existing = self.broker.positions.get(symbol)
                if existing:
                    r = self.broker.close_position(symbol, mid, reason="manual_sell")
                    self.save()
                    return {
                        "success": r.get("success", False),
                        "order_id": self._next_order_id(),
                        "filled": True,
                        "symbol": symbol,
                        "side": side,
                        "net_pnl": r.get("net_pnl"),
                        "paper": True,
                    }
                return {"success": False, "error": "short selling not implemented in paper mode"}

            portion = portion_of_balance
            if portion is None and size is not None and mid > 0:
                avail = self.broker.available_balance(self._marks())
                notional = size * mid
                portion = notional / avail if avail > 0 else 0.0

            r = self.broker.open_long(
                symbol, portion=portion or 0.0, mid_price=mid, leverage=leverage,
                take_profit=tp_price, stop_loss=sl_price,
            )

            if not r.get("success"):
                # 護欄擋下：對齊真實系統的 skipped 語意
                return {
                    "success": False,
                    "skipped": r.get("skipped", "rejected"),
                    "paper": True,
                    "symbol": symbol,
                }

            self.save()
            return {
                "success": True,
                "order_id": self._next_order_id(),
                "filled": True,
                "symbol": symbol,
                "side": side,
                "size": r["size"],
                "entry_price": r["entry_price"],
                "margin": r["margin"],
                "fee": r["fee"],
                "leverage": leverage,
                "paper": True,
            }

    def place_order_with_tpsl(
        self,
        db,
        symbol: str,
        is_buy: bool,
        size: Optional[float] = None,
        price: Optional[float] = None,
        leverage: int = 1,
        time_in_force: str = "Ioc",
        reduce_only: bool = False,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        tp_execution: str = "limit",
        sl_execution: str = "market",
        **kwargs,
    ) -> Dict[str, Any]:
        """模擬下單 + TP/SL。

        這是 program_execution_service 實際呼叫的方法（不是 place_order）。
        對齊真實 client 的回傳格式：{status, order_id, ...}。

        TP/SL 在 paper 模式不做成獨立的子單 —— 鏈上那樣做會有「主單成交但
        子單缺失」的風險（PRP-P2-4 記錄過）。這裡直接記進持倉，由每次
        check_exits() 用現價判定。這是刻意的取捨。
        """
        with self._lock:
            mid = price or self._mainnet_price(symbol)
            if mid is None:
                return {"status": "error", "error": f"no price for {symbol}"}

            if not is_buy:
                existing = self.broker.positions.get(symbol)
                if existing:
                    r = self.broker.close_position(symbol, mid, reason="reduce_only")
                    self.save()
                    return {
                        "status": "filled",
                        "order_id": self._next_order_id(),
                        "filled": True,
                        "symbol": symbol,
                        "avg_price": r.get("exit_price"),
                        "net_pnl": r.get("net_pnl"),
                        "paper": True,
                    }
                return {"status": "error", "error": "no position to reduce"}

            if reduce_only:
                return {"status": "error", "error": "reduce_only buy is meaningless"}

            # 執行服務傳入的是絕對 size，換算成 portion 交給 broker
            portion = None
            if size and mid > 0:
                marks = self._marks()
                avail = self.broker.available_balance(marks)
                portion = (size * mid) / avail if avail > 0 else 0.0

            r = self.broker.open_long(
                symbol, portion=portion or 0.0, mid_price=mid, leverage=leverage,
                take_profit=take_profit_price, stop_loss=stop_loss_price,
            )

            if not r.get("success"):
                reason = r.get("skipped", "rejected")
                # 風控跳過用 __skipped__ 標記，讓呼叫端能區分「跳過」與「失敗」
                if "cap" in reason or "cooldown" in reason or "max_positions" in reason \
                        or "insufficient" in reason or "available" in reason:
                    return {"__skipped__": True, "reason": f"skipped: {reason}", "detail": "", "paper": True}
                return {"status": "error", "error": reason, "paper": True}

            self.save()
            return {
                "status": "filled",
                "order_id": self._next_order_id(),
                "filled": True,
                "symbol": symbol,
                "size": r["size"],
                "avg_price": r["entry_price"],
                "margin": r["margin"],
                "fee": r["fee"],
                "leverage": leverage,
                # TP/SL 由本地 check_exits 判定，不產生鏈上子單
                "tp_order_id": None,
                "sl_order_id": None,
                "paper": True,
                "note": "TP/SL tracked locally, triggered by check_exits()",
            }

    def set_leverage(self, db, symbol: str, leverage: int) -> bool:
        """Paper 模式下槓桿在開倉時就固定了，這裡只記錄。"""
        return True

    def cancel_order(self, db, order_id: Any, symbol: str) -> bool:
        """Paper 模式沒有掛單可撤。"""
        return False

    # ---------------- paper 專屬 ----------------

    def check_exits(self) -> List[Dict[str, Any]]:
        """掃描 TP/SL。用 mainnet 現價判定，每次排程觸發時呼叫。"""
        with self._lock:
            marks = self._marks()
            if not marks:
                return []
            results = self.broker.check_exits(marks)
            if results:
                self.save()
                logger.info("[PaperClient] account %s closed %d position(s): %s",
                            self.account_id, len(results),
                            [(r["symbol"], r["reason"], r["net_pnl"]) for r in results])
            return results

    def performance(self) -> Dict[str, Any]:
        """績效摘要。欄位對齊 backtest Annual Gate 的需求。"""
        perf = self.broker.performance(self._marks())
        perf["mode"] = "paper"
        perf["data_source"] = "hyperliquid_mainnet"
        return perf

    def get_query_log(self) -> List[Dict[str, Any]]:
        return []

    def clear_query_log(self) -> None:
        pass

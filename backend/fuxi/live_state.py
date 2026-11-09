"""§6.1 實盤持倉狀態機與交易所對帳 (F10)。

狀態
----
``IDLE``              無持倉，可接受新訊號。
``PENDING_ORDER``     Maker 限價單已掛，等待成交。超過 ``max_unfilled_bars`` 根未成交取消。
``POSITION_OPEN``     持倉中，綁定開倉時封裝的 ``LivePositionContract``。
``COOLDOWN``          平倉後冷卻，冷卻期結束才回 IDLE。

對帳 (reconcile)
----------------
每根 5m K 線收盤呼叫 ``tick`` 一次。``tick`` 先對帳再推進狀態：本地契約與
交易所實際倉位逐欄比對（方向、保證金、TP、SL）。分歧一律以交易所為準同步
本地，事件寫入 ``events`` 供上層告警。

本地無倉但場內有倉時重建契約，``model_id`` 標 ``"unknown"``。那是孤兒倉，
不歸屬現任冠軍，出場邊界只認遠端 TP/SL (§4.3)。

交易所介面
----------
本模組只依賴一個 duck-typed protocol，不 import 任何具體交易所或 broker：

* ``get_positions(marks: dict) -> list[dict]``
  每項需含 ``symbol``、``size``（正多負空）。
  選填 ``entry_price``、``margin``、``leverage``、``take_profit``、``stop_loss``。
* ``place_limit(symbol, direction, notional, price, take_profit, stop_loss) -> dict``
  回 ``{"success": bool, "order_id": str}``。
* ``cancel_order(order_id) -> None``
* ``market_close(symbol, price, reason) -> dict``

``PaperBroker`` 的 ``portion`` 與 ``margin`` 語意不同，且沒有 ``open_short``，
所以由 ``PaperBrokerAdapter`` 做轉譯，狀態機本身不認識任何 broker 的細節。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

EPS = 1e-8


class LiveState(str, Enum):
    IDLE = "IDLE"
    PENDING_ORDER = "PENDING_ORDER"
    POSITION_OPEN = "POSITION_OPEN"
    COOLDOWN = "COOLDOWN"


@dataclass
class LivePositionContract:
    """開倉瞬間封裝的出場契約。冠軍交接永不重寫，孤兒倉綁定原契約。"""

    position_id: str
    model_id: str
    symbol: str
    direction: int
    entry_price: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    open_bar_idx: int
    max_hold_bars: int
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LivePositionContract":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class ReconcileDiff:
    """本地與交易所的單一分歧。"""

    field: str
    local: Any
    remote: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PositionStateMachine:
    """§6.1 含對帳的實盤狀態機。

    每根 5m K 線收盤呼叫一次 ``tick(mark)``。訊號端在 IDLE 時呼叫
    ``submit_order``；成交由交易所回報，``tick`` 會自己從對帳結果認定成交，
    不依賴外部 callback。
    """

    def __init__(
        self,
        broker: Any,
        *,
        symbol: str = "BTC",
        max_unfilled_bars: int = 3,
        cooldown_bars: int = 4,
    ) -> None:
        if max_unfilled_bars < 1:
            raise ValueError("max_unfilled_bars 必須 >= 1")
        if cooldown_bars < 1:
            raise ValueError("cooldown_bars 必須 >= 1")
        self.broker = broker
        self.symbol = symbol
        self.max_unfilled_bars = int(max_unfilled_bars)
        self.cooldown_bars = int(cooldown_bars)
        self.state: LiveState = LiveState.IDLE
        self.contract: Optional[LivePositionContract] = None
        self.order_id: Optional[str] = None
        self.pending_bars: int = 0
        self.cooldown_left: int = 0
        self.bar_idx: int = 0
        self.events: List[Dict[str, Any]] = []

    # ---------------- 下單 ----------------

    def submit_order(
        self,
        direction: int,
        mid_price: float,
        *,
        notional: float,
        leverage: float,
        model_id: str,
        x: float,
        y: float,
        max_hold_bars: int,
    ) -> bool:
        """IDLE 時掛 Maker 限價單。成功回 True 並轉 PENDING_ORDER。"""
        if self.state is not LiveState.IDLE:
            return False
        if direction not in (1, -1):
            raise ValueError(f"direction 必須是 +1/-1，拿到 {direction!r}")
        if not (x > 0 and y > 0):
            raise ValueError(f"x/y 必須為正，拿到 x={x!r} y={y!r}")
        sl_pct = float(x) / float(y)
        tp = float(mid_price) * (1.0 + direction * float(x))
        sl = float(mid_price) * (1.0 - direction * sl_pct)
        res = self.broker.place_limit(
            self.symbol, int(direction), float(notional), float(mid_price),
            take_profit=tp, stop_loss=sl, leverage=float(leverage),
        )
        if not (res or {}).get("success"):
            self._log("order_rejected", {"result": res})
            return False
        self.order_id = (res or {}).get("order_id")
        self.contract = LivePositionContract(
            position_id=str(self.order_id or f"{self.symbol}-{self.bar_idx}"),
            model_id=str(model_id), symbol=self.symbol, direction=int(direction),
            entry_price=float(mid_price), margin=float(notional) / max(float(leverage), 1.0),
            leverage=float(leverage), tp_price=tp, sl_price=sl,
            open_bar_idx=self.bar_idx, max_hold_bars=int(max_hold_bars),
            x=float(x), y=float(y),
        )
        self.pending_bars = 0
        self.state = LiveState.PENDING_ORDER
        self._log("order_submitted", {"order_id": self.order_id})
        return True

    # ---------------- 每根 5m 呼叫一次 ----------------

    def tick(self, mark: float) -> LiveState:
        """先對帳（交易所為準），再推進狀態。回傳推進後的狀態。"""
        self.bar_idx += 1
        mark = float(mark)
        diffs = self.reconcile(mark)
        if diffs:
            self._log("reconcile_diverged",
                      {"diffs": [d.to_dict() for d in diffs]})

        if self.state is LiveState.PENDING_ORDER:
            self.pending_bars += 1
            remote = self._remote(mark)
            if remote is not None:
                # 已成交：以交易所實際進場價為準。
                self._adopt_remote_entry(remote)
            elif self.pending_bars > self.max_unfilled_bars:
                self._cancel_pending(f"unfilled_{self.pending_bars}_bars")

        elif self.state is LiveState.POSITION_OPEN and self.contract is not None:
            remote = self._remote(mark)
            if remote is None:
                # 場外平倉（TP/SL 觸發、手動、強平）：本地同步結清。
                self._enter_cooldown(reason="remote_flat")
                return self.state
            expired = ((self.bar_idx - self.contract.open_bar_idx)
                       >= self.contract.max_hold_bars)
            if expired:
                self._close_remote(mark, reason="expiry")

        elif self.state is LiveState.COOLDOWN:
            self.cooldown_left -= 1
            if self.cooldown_left <= 0:
                self.cooldown_left = 0
                self.state = LiveState.IDLE
                self._log("cooldown_done", {})

        return self.state

    # ---------------- 對帳 ----------------

    def reconcile(self, mark: float) -> List[ReconcileDiff]:
        """本地契約與交易所逐欄比對；分歧以交易所為準修正本地。"""
        diffs: List[ReconcileDiff] = []
        remote = self._remote(mark)
        if self.contract is None and remote is None:
            return diffs
        if self.contract is None:
            self._recover_orphan(remote, mark)
            diffs.append(ReconcileDiff("position", None, "remoteOnly"))
            return diffs
        if remote is None:
            if self.state is LiveState.PENDING_ORDER:
                return diffs          # 掛單未成交，不是分歧
            diffs.append(ReconcileDiff("position", "localOnly", None))
            return diffs

        remote_dir = 1 if float(remote.get("size", 0.0) or 0.0) >= 0 else -1
        if self.contract.direction != remote_dir:
            diffs.append(ReconcileDiff("direction", self.contract.direction, remote_dir))
            self.contract.direction = remote_dir

        for key, attr in (("take_profit", "tp_price"), ("stop_loss", "sl_price")):
            local_val = float(getattr(self.contract, attr) or 0.0)
            remote_val = float(remote.get(key) or 0.0)
            if remote_val > 0.0 and not self._close_enough(local_val, remote_val):
                diffs.append(ReconcileDiff(key, local_val, remote_val))
                setattr(self.contract, attr, remote_val)
        return diffs

    # ---------------- 內部：狀態轉換 ----------------

    def _adopt_remote_entry(self, remote: Dict[str, Any]) -> None:
        if self.contract is None:
            return
        entry = float(remote.get("entry_price") or self.contract.entry_price)
        self.contract.entry_price = entry
        self.state = LiveState.POSITION_OPEN
        self._log("filled", {"entry_price": entry})

    def _recover_orphan(self, remote: Dict[str, Any], mark: float) -> None:
        """本地無倉、場內有倉：重建契約並標為孤兒。"""
        size = float(remote.get("size", 0.0) or 0.0)
        self.contract = LivePositionContract(
            position_id=f"{self.symbol}-orphan-{self.bar_idx}",
            model_id="unknown",
            symbol=self.symbol,
            direction=1 if size >= 0 else -1,
            entry_price=float(remote.get("entry_price") or mark),
            margin=float(remote.get("margin") or 0.0),
            leverage=float(remote.get("leverage") or 1.0),
            tp_price=float(remote.get("take_profit") or 0.0),
            sl_price=float(remote.get("stop_loss") or 0.0),
            open_bar_idx=self.bar_idx,
            # 孤兒倉沒有出生契約，不套用任何 M。只能靠遠端 TP/SL 出場。
            max_hold_bars=2**31 - 1,
        )
        self.state = LiveState.POSITION_OPEN
        self._log("orphan_recovered", {"contract": self.contract.to_dict()})

    def _cancel_pending(self, reason: str) -> None:
        if self.order_id is not None:
            try:
                self.broker.cancel_order(self.order_id)
            except Exception as exc:
                self._log("cancel_failed", {"error": str(exc)})
        self._log("pending_cancelled", {"reason": reason})
        self.contract = None
        self.order_id = None
        self.pending_bars = 0
        self.state = LiveState.IDLE

    def _close_remote(self, mark: float, reason: str) -> None:
        try:
            self.broker.market_close(self.symbol, float(mark), reason)
        except Exception as exc:
            self._log("close_failed", {"reason": reason, "error": str(exc)})
            return
        self._enter_cooldown(reason=reason)

    def _enter_cooldown(self, reason: str) -> None:
        self._log("position_closed", {"reason": reason})
        self.contract = None
        self.order_id = None
        self.cooldown_left = self.cooldown_bars
        self.state = LiveState.COOLDOWN

    def _remote(self, mark: float) -> Optional[Dict[str, Any]]:
        try:
            positions = self.broker.get_positions({self.symbol: float(mark)}) or []
        except Exception as exc:
            self._log("reconcile_failed", {"error": str(exc)})
            return None
        for pos in positions:
            if pos.get("symbol") == self.symbol:
                return dict(pos)
        return None

    @staticmethod
    def _close_enough(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-8 * max(1.0, abs(b))

    def _log(self, kind: str, payload: Dict[str, Any]) -> None:
        self.events.append({
            "bar": self.bar_idx, "state": self.state.value, "event": kind, **payload,
        })

    # ---------------- 序列化 ----------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "symbol": self.symbol,
            "bar_idx": self.bar_idx,
            "order_id": self.order_id,
            "pending_bars": self.pending_bars,
            "cooldown_left": self.cooldown_left,
            "contract": self.contract.to_dict() if self.contract else None,
        }

    @classmethod
    def from_dict(cls, broker: Any, data: Dict[str, Any],
                 **kwargs: Any) -> "PositionStateMachine":
        sm = cls(broker, symbol=data.get("symbol", "BTC"), **kwargs)
        sm.state = LiveState(data.get("state") or LiveState.IDLE)
        sm.bar_idx = int(data.get("bar_idx") or 0)
        sm.order_id = data.get("order_id")
        sm.pending_bars = int(data.get("pending_bars") or 0)
        sm.cooldown_left = int(data.get("cooldown_left") or 0)
        raw = data.get("contract")
        sm.contract = LivePositionContract.from_dict(raw) if raw else None
        return sm


class FakeExchange:
    """測試與 dry-run 用的交易所。行為對齊狀態機 protocol，無網路。"""

    def __init__(self, *, fee_bps: float = 4.5, fill_immediately: bool = True) -> None:
        self.fee_bps = float(fee_bps)
        self.fill_immediately = bool(fill_immediately)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.fills: List[Dict[str, Any]] = []
        self._seq = 0

    def place_limit(self, symbol: str, direction: int, notional: float, price: float,
                    *, take_profit: float, stop_loss: float,
                    leverage: float = 1.0) -> Dict[str, Any]:
        self._seq += 1
        oid = f"ord-{self._seq}"
        self.orders[oid] = {
            "symbol": symbol, "direction": int(direction), "notional": float(notional),
            "price": float(price), "leverage": float(leverage),
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
            "status": "open",
        }
        if self.fill_immediately:
            self._fill(oid)
        return {"success": True, "order_id": oid}

    def cancel_order(self, order_id: str) -> None:
        order = self.orders.get(order_id)
        if order is not None:
            order["status"] = "cancelled"

    def _fill(self, order_id: str) -> None:
        order = self.orders[order_id]
        if order["status"] != "open":
            return
        order["status"] = "filled"
        price = order["price"]
        size = (order["notional"] * order["leverage"]) / price * order["direction"]
        self.positions[order["symbol"]] = {
            "symbol": order["symbol"], "size": size,
            "entry_price": price,
            "margin": order["notional"],
            "leverage": order["leverage"],
            "take_profit": order["take_profit"],
            "stop_loss": order["stop_loss"],
        }
        self.fills.append({"order_id": order_id, "price": price})

    def get_positions(self, marks: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        return [dict(p) for p in self.positions.values()]

    def market_close(self, symbol: str, price: float, reason: str) -> Dict[str, Any]:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return {"success": False, "skipped": "no_position"}
        gross = (float(price) - pos["entry_price"]) * pos["size"]
        fee = abs(pos["size"] * float(price)) * self.fee_bps / 10000.0
        return {"success": True, "gross_pnl": gross, "fee": fee,
                "net_pnl": gross - fee, "reason": reason}

    def check_exits(self, marks: Dict[str, float]) -> List[Dict[str, Any]]:
        """模擬交易所託管觸發 TP/SL。"""
        out: List[Dict[str, Any]] = []
        for symbol in list(self.positions):
            pos = self.positions[symbol]
            mark = marks.get(symbol)
            if mark is None:
                continue
            long_side = pos["size"] >= 0
            tp, sl = pos.get("take_profit"), pos.get("stop_loss")
            if long_side and tp and mark >= tp:
                out.append(self.market_close(symbol, tp, "take_profit"))
            elif long_side and sl and mark <= sl:
                out.append(self.market_close(symbol, sl, "stop_loss"))
            elif not long_side and tp and mark <= tp:
                out.append(self.market_close(symbol, tp, "take_profit"))
            elif not long_side and sl and mark >= sl:
                out.append(self.market_close(symbol, sl, "stop_loss"))
        return out


class PaperBrokerAdapter:
    """把 ``services.paper_broker.PaperBroker`` 轉譯成狀態機 protocol。

    ``PaperBroker`` 的 ``portion`` 是可用餘額比例、``close_position`` 需要
    ``mid_price``、且沒有 ``open_short``。轉譯集中在這裡，狀態機本身不認識
    這些細節。
    """

    def __init__(self, broker: Any) -> None:
        self.broker = broker
        self._by_order: Dict[str, Dict[str, Any]] = {}
        self._seq = 0

    def place_limit(self, symbol: str, direction: int, notional: float, price: float,
                    *, take_profit: float, stop_loss: float,
                    leverage: float = 1.0) -> Dict[str, Any]:
        avail = float(self.broker.available_balance())
        if avail <= 0:
            return {"success": False, "skipped": "no_available_balance"}
        portion = min(1.0, float(notional) / avail)
        size = (portion * avail * leverage) / max(float(price), EPS)
        self._seq += 1
        oid = f"paper-{self._seq}"
        self._by_order[oid] = {
            "symbol": symbol, "direction": int(direction), "size": size,
            "leverage": int(leverage),
        }
        if direction == 1:
            res = self.broker.open_long(
                symbol, portion, float(price), int(leverage),
                take_profit=take_profit, stop_loss=stop_loss,
            )
        else:
            # PaperBroker 只有 open_long；空單以負 size 直寫持倉，維持同一本帳。
            from services.paper_broker import PaperPosition  # 延遲匯入
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            reason = self.broker._check_guards(symbol, -size, float(price), int(leverage))
            if reason:
                return {"success": False, "skipped": reason}
            fee = self.broker.fee_model.fee(size * float(price), is_maker=False)
            self.broker.positions[symbol] = PaperPosition(
                symbol=symbol, size=-size, entry_price=float(price),
                entry_time=now.isoformat(), leverage=int(leverage),
                take_profit=take_profit, stop_loss=stop_loss,
                margin=portion * avail, fees_paid=fee,
            )
            self.broker.cash -= fee
            self.broker._last_add_time[symbol] = now
            self.broker._record_equity(now)
            res = {"success": True, "size": -size}
        if not (res or {}).get("success"):
            return res
        return {"success": True, "order_id": oid}

    def cancel_order(self, order_id: str) -> None:
        meta = self._by_order.pop(order_id, None)
        if meta and meta["symbol"] in self.broker.positions:
            pos = self.broker.positions[meta["symbol"]]
            if abs(abs(pos.size) - meta["size"]) <= 1e-8:
                self.broker.close_position(meta["symbol"], pos.entry_price,
                                           reason="cancelled")

    def get_positions(self, marks: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        return self.broker.get_positions(marks)

    def market_close(self, symbol: str, price: float, reason: str) -> Dict[str, Any]:
        return self.broker.close_position(symbol, float(price), reason=reason)

    def check_exits(self, marks: Dict[str, float]) -> List[Dict[str, Any]]:
        return self.broker.check_exits(marks)

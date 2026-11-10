"""F10 §6.1 實盤持倉狀態機與交易所對帳的回歸測試。

約定：每個用例都在 FakeExchange 上跑完整回合（掛單 → 成交 → 出場），
不用任何網路或資料庫。
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fuxi.live_state import FakeExchange, LiveState, PositionStateMachine


def _open(machine: PositionStateMachine, **kw):
    args = dict(direction=1, mid_price=100.0, notional=100.0, leverage=2.0,
                model_id="FUXI_C0001", x=0.03, y=2.0, max_hold_bars=144)
    args.update(kw)
    assert machine.submit_order(**args) is True


def test_full_cycle_fill_tp():
    ex = FakeExchange(fill_immediately=True)
    sm = PositionStateMachine(ex, symbol="BTC", cooldown_bars=2)
    assert sm.state is LiveState.IDLE
    _open(sm)
    assert sm.state is LiveState.PENDING_ORDER
    assert sm.tick(100.5) is LiveState.POSITION_OPEN
    # TP 在 103.0，mark 觸發交易所託管出場前先由本地看到遠端平倉
    ex.check_exits({"BTC": 103.5})
    assert sm.tick(103.5) is LiveState.COOLDOWN
    assert sm.tick(103.0) is LiveState.COOLDOWN
    assert sm.tick(103.0) is LiveState.IDLE


def test_pending_timeout_cancels():
    ex = FakeExchange(fill_immediately=False)
    sm = PositionStateMachine(ex, symbol="BTC", max_unfilled_bars=2)
    _open(sm)
    assert sm.tick(100.1) is LiveState.PENDING_ORDER
    assert sm.tick(100.1) is LiveState.PENDING_ORDER
    # 第 3 根超過上限：取消並回 IDLE
    assert sm.tick(100.1) is LiveState.IDLE
    assert sm.contract is None
    assert ex.orders["ord-1"]["status"] == "cancelled"


def test_expiry_closes_position():
    ex = FakeExchange(fill_immediately=True)
    sm = PositionStateMachine(ex, symbol="BTC")
    _open(sm, max_hold_bars=2)
    assert sm.tick(100.5) is LiveState.POSITION_OPEN
    # 到期：本地發市價出場
    assert sm.tick(100.6) is LiveState.COOLDOWN
    assert "BTC" not in ex.positions


def test_remote_flat_syncs_to_cooldown():
    ex = FakeExchange(fill_immediately=True)
    sm = PositionStateMachine(ex, symbol="BTC")
    _open(sm)
    assert sm.tick(100.5) is LiveState.POSITION_OPEN
    # 場外手動平倉：本地同步結清，不重建
    ex.positions.clear()
    assert sm.tick(100.5) is LiveState.COOLDOWN


def test_reconcile_direction_divergence_adopts_remote():
    ex = FakeExchange(fill_immediately=True)
    sm = PositionStateMachine(ex, symbol="BTC")
    _open(sm, direction=1)
    assert sm.tick(100.5) is LiveState.POSITION_OPEN
    # 場外反手：本地方向以交易所為準
    pos = ex.positions["BTC"]
    pos["size"] = -abs(pos["size"])
    diffs = sm.reconcile(100.5)
    assert any(d.field == "direction" for d in diffs)
    assert sm.contract is not None and sm.contract.direction == -1


def test_orphan_recovery_and_serialization():
    ex = FakeExchange(fill_immediately=False)
    ex.positions["BTC"] = {
        "symbol": "BTC", "size": 1.5, "entry_price": 99.0,
        "margin": 50.0, "leverage": 2, "take_profit": 102.0, "stop_loss": 97.0,
    }
    sm = PositionStateMachine(ex, symbol="BTC")
    diffs = sm.reconcile(100.0)
    assert any(d.field == "position" for d in diffs)
    assert sm.state is LiveState.POSITION_OPEN
    assert sm.contract is not None and sm.contract.model_id == "unknown"
    clone = PositionStateMachine.from_dict(ex, sm.to_dict())
    assert clone.state is LiveState.POSITION_OPEN
    assert clone.contract is not None and clone.contract.tp_price == 102.0


def test_submit_rejected_outside_idle():
    ex = FakeExchange(fill_immediately=True)
    sm = PositionStateMachine(ex, symbol="BTC")
    _open(sm)
    # PENDING 時再掛單：拒絕
    assert sm.submit_order(direction=-1, mid_price=100.0, notional=50.0,
                           leverage=1.0, model_id="X",
                           x=0.03, y=2.0, max_hold_bars=10) is False

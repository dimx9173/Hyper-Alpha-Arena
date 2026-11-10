"""Cross-path fee equivalence: paper broker vs backtest simulator.

同一個 FeeSchedule 物件餵給兩條路徑，對 maker / taker 範例算出相同費用。
"""

import pytest

from backtest.execution_simulator import ExecutionSimulator
from config.fees import FeeSchedule
from services.paper_broker import FeeModel


def test_same_schedule_same_fee_maker_and_taker():
    sched = FeeSchedule(maker_fee_bps=1.5, taker_fee_bps=4.5, slippage_bps=2.0)
    broker_fee = FeeModel(
        maker_fee_bps=sched.maker_fee_bps,
        taker_fee_bps=sched.taker_fee_bps,
        slippage_bps=sched.slippage_bps,
    )
    sim = ExecutionSimulator(fee_schedule=sched)

    notional = 10_000.0
    # maker
    assert broker_fee.fee(notional, is_maker=True) == sim.calculate_fee(notional, is_maker=True) == 1.5
    # taker
    assert broker_fee.fee(notional, is_maker=False) == sim.calculate_fee(notional, is_maker=False) == 4.5


def test_fee_model_is_schedule_alias_with_same_defaults():
    default_sched = FeeSchedule()
    default_model = FeeModel()
    assert isinstance(default_model, FeeSchedule)
    assert (default_model.maker_fee_bps, default_model.taker_fee_bps, default_model.slippage_bps) == \
        (default_sched.maker_fee_bps, default_sched.taker_fee_bps, default_sched.slippage_bps)

    notional = 50_000.0
    assert default_model.fee(notional) == default_sched.fee(notional)
    assert default_model.entry_price(100.0, is_buy=True) == default_sched.entry_price(100.0, is_buy=True)
    assert default_model.exit_price(100.0, is_buy=True) == default_sched.exit_price(100.0, is_buy=True)


def test_percent_boundary_conversion_once():
    # 邊界換算：0.035% taker / 0.02% maker / 0.05% slip → bps 只換算一次
    sim = ExecutionSimulator(slippage_percent=0.05, fee_rate=0.035, maker_fee_rate=0.02)
    assert sim.fees.taker_fee_bps == pytest.approx(3.5)
    assert sim.fees.maker_fee_bps == pytest.approx(2.0)
    assert sim.fees.slippage_bps == pytest.approx(5.0)
    direct = ExecutionSimulator(fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=3.5, slippage_bps=5.0))
    assert sim.calculate_fee(10_000.0) == pytest.approx(direct.calculate_fee(10_000.0))

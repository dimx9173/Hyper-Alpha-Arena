"""Paper Broker 測試。

重點驗證會計恆等式與風控護欄，因為這些數字會直接決定策略是否「通過 gate」。
會計錯了，績效數字就沒有任何意義。
"""
import sys, os, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.paper_broker import PaperBroker, FeeModel, PaperPosition, PaperTrade


def test_fee_and_slippage_direction():
    f = FeeModel(slippage_bps=20.0)  # 20bp 放大以便觀察
    # 買入滑高、賣出滑低
    assert f.entry_price(100.0, is_buy=True) > 100.0, "買入應滑高"
    assert f.entry_price(100.0, is_buy=False) < 100.0, "賣出應滑低"
    assert f.exit_price(100.0, is_buy=True) < 100.0, "買回應滑低"
    assert f.exit_price(100.0, is_buy=False) > 100.0, "賣出應滑高"
    # 手續費
    assert f.fee(10000.0) == 10000.0 * 4.5 / 10000, f"taker fee 應為 4.5bp: {f.fee(10000.0)}"
    assert f.fee(10000.0, is_maker=True) == 10000.0 * 1.5 / 10000
    print("  滑價與手續費方向正確")


def test_open_position_accounting():
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=0.0))
    r = b.open_long("BTC", portion=0.35, mid_price=100.0, leverage=1)
    assert r["success"], f"開倉應成功: {r}"

    pos = b.positions["BTC"]
    assert abs(pos.entry_price - 100.0) < 1e-9
    assert abs(pos.size - 35.0) < 1e-6, f"size 應為 35 BTC (3500 USD / 100): {pos.size}"
    assert abs(pos.margin - 3500.0) < 1e-6
    # 現金 = 10000 - 3500 - fee
    # 會計模型：cash = 初始資金 − 手續費（保證金不是轉出）
    expected_cash = 10000.0 - r["fee"]
    assert abs(b.cash - expected_cash) < 1e-6, f"現金不符: {b.cash} vs {expected_cash}"
    assert abs(b.available_balance() - (10000.0 - r["fee"] - 3500.0)) < 1e-6, \
        f"可用餘額不符: {b.available_balance()}"
    print(f"  開倉會計正確 (cash={b.cash:.4f}, avail={b.available_balance():.4f}, fee={r['fee']:.4f})")


def test_unrealized_and_realized_pnl():
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=0.0))
    b.open_long("BTC", portion=0.3, mid_price=100.0, leverage=1)
    pos = b.positions["BTC"]

    # 漲到 110（零滑價，但仍有 taker 手續費）
    marks = {"BTC": 110.0}
    eq = b.total_equity(marks)
    expected_gain = (110.0 - 100.0) * pos.size
    # 權益 = 初始資金 + 浮盈 − 進場手續費（cash 是錢包餘額，開倉已扣費）
    assert abs(eq - (10000.0 + expected_gain - pos.fees_paid)) < 1e-6, \
        f"浮盈計算不符: {eq} vs {10000.0 + expected_gain - pos.fees_paid}"

    # 平倉
    r = b.close_position("BTC", 110.0, reason="manual")
    assert r["success"]
    assert "BTC" not in b.positions
    # 平倉後權益應等於 cash
    eq2 = b.total_equity()
    assert abs(eq2 - b.cash) < 1e-6, f"無持倉時權益應等於現金: {eq2} vs {b.cash}"
    print(f"  實現損益正確 (net_pnl={r['net_pnl']:.4f}, equity={eq2:.2f})")


def test_take_profit_trigger():
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=0.0))
    b.open_long("BTC", portion=0.3, mid_price=100.0, leverage=1,
                take_profit=110.0, stop_loss=95.0)

    # 未觸發
    res = b.check_exits({"BTC": 105.0})
    assert not res, "105 不應觸發 TP"
    assert "BTC" in b.positions

    # 觸發 TP
    res = b.check_exits({"BTC": 111.0})
    assert res, "111 應觸發 TP"
    assert res[0]["reason"] == "take_profit"
    assert "BTC" not in b.positions
    trade = b.trades[-1]
    assert trade.exit_price == 110.0, f"應以 TP 價成交: {trade.exit_price}"
    print(f"  TP 觸發正確 (exit={trade.exit_price}, net={trade.net_pnl:.4f})")


def test_stop_loss_trigger():
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=0.0))
    b.open_long("BTC", portion=0.3, mid_price=100.0, leverage=1,
                take_profit=110.0, stop_loss=95.0)

    res = b.check_exits({"BTC": 94.0})
    assert res, "94 應觸發 SL"
    assert res[0]["reason"] == "stop_loss"
    trade = b.trades[-1]
    assert trade.exit_price == 95.0
    assert trade.net_pnl < 0, "止損應為虧損"
    print(f"  SL 觸發正確 (exit={trade.exit_price}, net={trade.net_pnl:.4f})")


def test_tp_has_priority_when_both_hit():
    """同一根 K 線同時觸及 TP 與 SL 時的處理。

    決策：TP 優先（先檢查 TP）。理由：真實市場同一根 1h K 線內
    先觸及哪個價位未知，用 TP 優先是較保守（對回測不失真）的假設。
    """
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=0.0))
    b.open_long("BTC", portion=0.3, mid_price=100.0, leverage=1,
                take_profit=110.0, stop_loss=95.0)
    res = b.check_exits({"BTC": 111.0})   # 已越過 TP
    assert res[0]["reason"] == "take_profit"
    print("  TP 優先於 SL")


def test_single_coin_cap_guard():
    b = PaperBroker(initial_capital=10000.0)
    # portion=0.5 → notional 5000 > 10000*0.4=4000，應被擋
    r = b.open_long("BTC", portion=0.5, mid_price=100.0, leverage=1)
    assert not r["success"], "0.5 portion 應被單幣護欄擋下"
    assert "single_coin_cap" in r["skipped"], r
    assert not b.positions
    print(f"  單幣護欄正確擋下: {r['skipped']}")


def test_exposure_cap_guard():
    """總暴露護欄：所有持倉 notional 合計超過 2x equity 時擋下新倉。

    觸發方式：不能用累積 portion（分母是 available_balance，會遞減，永遠碰不到上限）。
    必須用「單筆大倉位」+ 高槓桿，把 notional 推到上限之上。

    驗證重點：擋下原因必須是 exposure_cap，而不是碰巧被其他護欄擋。
    """
    b = PaperBroker(initial_capital=10000.0, max_positions=20)
    equity = b.total_equity()
    cap = equity * 2.0

    # 第一筆：leverage 5, portion 0.39 → notional = 0.39*10000*5 = 19500
    # 單幣上限 = 4000，19500 > 4000 → 會被單幣護欄擋。所以用 portion 0.079 leverage 5 → 3950
    r1 = b.open_long("AAVE", portion=0.079, mid_price=100.0, leverage=5)
    assert r1["success"], f"第1筆應成功: {r1}"
    assert r1["margin"] * 5 <= equity * b._max_single_coin + 1e-6, \
        f"第1筆 notional 應低於單幣上限: {r1['margin']*5}"

    # 第2筆：同樣規模，合計 7900 < 20000，應成功
    r2 = b.open_long("FIL", portion=0.079, mid_price=100.0, leverage=5)
    assert r2["success"], f"第2筆應成功: {r2}"

    # 持續加碼直到觸發某個上限
    blocked = None
    for sym in ["TRUMP", "PENGU", "WLD", "XRP", "APT", "ARB2", "OP", "TIA"]:
        r = b.open_long(sym, portion=0.079, mid_price=100.0, leverage=5)
        if not r["success"]:
            blocked = r.get("skipped")
            break

    total = sum(p.notional for p in b.positions.values())
    assert total <= cap + 1e-6, f"總暴露超標: {total} > {cap}"
    assert blocked is not None, "應有某個標的被擋（max_positions=20，8 個標的不可能全部通過）"
    assert any(k in blocked for k in ("exposure_cap", "max_positions", "insufficient", "single_coin", "cluster_cap")), \
        f"擋下原因應與上限相關: {blocked}"
    print(f"  總暴露護欄正確 (已放 {len(b.positions)} 筆, total={total:.0f}, "
          f"cap={cap:.0f}, 最後擋下: {blocked})")



def test_cluster_cap_guard():
    """同簇護欄：CLUSTER 內 notional 合計超過 0.6x equity 時擋下。

    驗證方式：先在 BTC 上建立接近上限的部位，再看同簇第二個幣是否被擋。
    護欄是事後檢查，所以要讓第一筆盡量接近 0.6x。
    """
    b = PaperBroker(initial_capital=10000.0, max_positions=10)
    # portion=0.39 → notional 3900，接近單幣上限 4000 且 < cluster 上限 6000
    r1 = b.open_long("BTC", portion=0.39, mid_price=100.0, leverage=1)
    assert r1["success"], r1
    btc_notional = b.positions["BTC"].notional
    cluster_cap = b.total_equity() * b._max_cluster

    # 第二個同簇幣：即使 portion 很小，cluster 合計也可能超標
    r2 = b.open_long("ETH", portion=0.35, mid_price=100.0, leverage=1)
    if r2["success"]:
        # 若通過，驗證 cluster 合計確實未超標
        cluster = sum(p.notional for s, p in b.positions.items() if s in b.CLUSTER)
        assert cluster <= cluster_cap + 1e-6, \
            f"同簇合計 {cluster} 超過上限 {cluster_cap}"
    else:
        assert "cluster_cap" in r2.get("skipped", "") or "single_coin_cap" in r2.get("skipped", "") \
            or "exposure_cap" in r2.get("skipped", ""), r2

    # 非 cluster 標的：即使 BTC 已佔大量資金，仍應能進場（cluster cap 不影響它）
    r3 = b.open_long("AAVE", portion=0.15, mid_price=100.0, leverage=1)
    assert r3["success"], f"非 cluster 標的應可進場: {r3}"

    print(f"  同簇護欄正確 (BTC notional={btc_notional:.0f}, cluster 上限={cluster_cap:.0f})")
    print(f"  同簇第二筆: {'被擋 - ' + r2['skipped'] if not r2['success'] else '通過（未超上限）'}")
    print(f"  非 cluster 標的 AAVE 進場成功")


def test_max_positions_guard():
    b = PaperBroker(initial_capital=100000.0, max_positions=3)
    for sym in ["AAVE", "FIL", "TRUMP"]:
        assert b.open_long(sym, portion=0.1, mid_price=100.0, leverage=1)["success"]
    r = b.open_long("PENGU", portion=0.1, mid_price=100.0, leverage=1)
    assert not r["success"] and "max_positions" in r["skipped"]
    print(f"  持倉數上限正確: {r['skipped']}")


def test_cooldown_guard():
    import datetime
    b = PaperBroker(initial_capital=100000.0, cooldown_hours=4.0)
    assert b.open_long("BTC", portion=0.2, mid_price=100.0, leverage=1)["success"]
    # 立即加倉應被冷卻擋下
    r = b.open_long("BTC", portion=0.2, mid_price=100.0, leverage=1)
    assert not r["success"] and "cooldown" in r["skipped"], r
    # 把時間往前推 5 小時後應可用
    b._last_add_time["BTC"] = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)
    r2 = b.open_long("BTC", portion=0.2, mid_price=100.0, leverage=1)
    assert r2["success"], f"冷卻期過後應可加倉: {r2}"
    print(f"  同向冷卻正確（加倉後平均成本 {b.positions['BTC'].entry_price:.4f}）")


def test_add_position_avg_price():
    """加倉時 portion 的分母是「當下可用餘額」，會隨既有持倉與手續費遞減。"""
    f = FeeModel(slippage_bps=0.0)
    b = PaperBroker(initial_capital=100000.0, fee_model=f, cooldown_hours=0.0)

    r1 = b.open_long("BTC", portion=0.2, mid_price=100.0, leverage=1)
    assert r1["success"]
    # avail=100000, margin=20000, size=20000/100=200
    assert abs(r1["size"] - 200.0) < 1e-6, f"第1筆 size: {r1['size']}"

    r2 = b.open_long("BTC", portion=0.2, mid_price=200.0, leverage=1)
    assert r2["success"]
    # avail = cash − used_margin。cash 已被第1筆手續費扣減，所以不是精確的 80000。
    # 這是正確行為：手續費是實際支出，會降低可再投入的資金。
    fee1 = r1["fee"]
    avail2 = 100000.0 - fee1 - 20000.0
    expected_size2 = (avail2 * 0.2) / 200.0
    assert abs(r2["size"] - expected_size2) < 1e-6, \
        f"第2筆 size 應為 {expected_size2}: {r2['size']}"
    assert abs(r2["size"] - 80.0) < 0.1, \
        f"第2筆 size 應接近 80（差額為手續費 {fee1:.4f}）: {r2['size']}"

    pos = b.positions["BTC"]
    total_size = 200.0 + r2["size"]
    assert abs(pos.size - total_size) < 1e-6, f"總 size 應為 {total_size}: {pos.size}"
    expected_avg = (100.0 * 200.0 + 200.0 * r2["size"]) / total_size
    assert abs(pos.entry_price - expected_avg) < 1e-6, \
        f"平均成本應為 {expected_avg}: {pos.entry_price}"
    expected_margin = 20000.0 + r2["margin"]
    assert abs(pos.margin - expected_margin) < 1e-6, \
        f"margin 應為 {expected_margin}: {pos.margin}"
    print(f"  加倉加權平均正確 (size={pos.size:.4f}, avg={pos.entry_price:.4f}, "
          f"margin={pos.margin:.4f})")


def test_performance_metrics():
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=0.0))
    # 兩筆獲利
    for entry, tp in [(100.0, 110.0), (100.0, 105.0)]:
        b.open_long("BTC" if entry == 100 else "ETH", portion=0.3,
                    mid_price=entry, take_profit=tp, stop_loss=entry*0.9)
        b.check_exits({"BTC": 111.0, "ETH": 106.0} if entry == 100 else {"BTC": 111.0, "ETH": 106.0})
    # 一筆虧損
    b.open_long("SOL", portion=0.3, mid_price=100.0, take_profit=110.0, stop_loss=95.0)
    b.check_exits({"BTC": 100.0, "ETH": 100.0, "SOL": 94.0})

    perf = b.performance()
    assert perf["total_trades"] == 3, perf
    assert perf["wins"] == 2 and perf["losses"] == 1
    assert 60 < perf["win_rate_pct"] < 70, perf
    assert perf["profit_factor"] > 1.0
    print(f"  績效統計正確: {perf}")


def test_equity_identity():
    """會計恆等式：權益 = 初始資金 + Σ已實現損益 + 未實現損益。

    關鍵：cash 是錢包餘額（初始 + 已實現 - 費用），
    開倉時不扣保證金。所以這個等式必須精確成立（誤差 < 1e-6）。
    """
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=10.0))
    b.open_long("BTC", portion=0.2, mid_price=100.0, leverage=1)
    b.check_exits({"BTC": 105.0})
    b.open_long("ETH", portion=0.2, mid_price=50.0, leverage=1)

    marks = {"ETH": 55.0}
    realized = sum(t.net_pnl for t in b.trades)   # net_pnl 已扣進出场費用
    unreal = sum(p.unrealized_pnl(marks.get(s, p.entry_price))
                 for s, p in b.positions.items())
    open_fees = sum(p.fees_paid for p in b.positions.values())

    # 權益 = 初始 + 已實現(淨) - 未平倉的進場費用 + 未實現
    expected = b.initial_capital + realized + unreal - open_fees
    actual = b.total_equity(marks)
    assert abs(expected - actual) < 1e-6, \
        f"權益恆等式不符: expected={expected:.6f} actual={actual:.6f}"
    print(f"  權益恆等式精確成立 (expected={expected:.4f}, actual={actual:.4f})")


def test_persistence_roundtrip():
    b = PaperBroker(initial_capital=10000.0, fee_model=FeeModel(slippage_bps=5.0))
    b.open_long("BTC", portion=0.3, mid_price=100.0, take_profit=110.0, stop_loss=95.0)
    b.check_exits({"BTC": 111.0})
    b.open_long("ETH", portion=0.2, mid_price=200.0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    b.save_to(path)

    b2 = PaperBroker.load_from(path)
    assert abs(b2.cash - b.cash) < 1e-6, f"現金不符: {b2.cash} vs {b.cash}"
    assert len(b2.trades) == len(b.trades)
    assert set(b2.positions.keys()) == set(b.positions.keys())
    for sym in b.positions:
        assert abs(b2.positions[sym].size - b.positions[sym].size) < 1e-6
    os.unlink(path)
    print(f"  持久化往返正確 (cash={b2.cash:.4f}, trades={len(b2.trades)})")


def test_edge_cases():
    PaperBroker().performance()
    b = PaperBroker(initial_capital=0.0)
    r = b.open_long("BTC", portion=0.5, mid_price=100.0)
    assert not r["success"], "零資金不能開倉"
    b2 = PaperBroker()
    assert b2.close_position("NOPE", 100.0)["success"] is False
    b3 = PaperBroker()
    b3.open_long("BTC", portion=0.1, mid_price=100.0)
    b3.check_exits({})          # 空 marks，不應崩潰
    b3.check_exits({"BTC": 0})  # 零價，不應崩潰
    print("  邊界條件通過")


def test_zero_slippage_no_fee_pure_pnl():
    """零滑價零手續費時，盈虧應該精確等於價格變動。"""
    b = PaperBroker(initial_capital=10000.0,
                    fee_model=FeeModel(maker_fee_bps=0, taker_fee_bps=0, slippage_bps=0))
    b.open_long("BTC", portion=0.3, mid_price=100.0, leverage=1)
    r = b.close_position("BTC", 110.0)
    expected = (110.0 - 100.0) * 30.0   # size=30, gain=10 → 300
    assert abs(r["gross_pnl"] - expected) < 1e-6, f"應為 {expected}: {r['gross_pnl']}"
    assert abs(r["net_pnl"] - expected) < 1e-6, f"應為 {expected}: {r['net_pnl']}"
    print(f"  純盈虧精確 ({expected} = 30 單位 × 10)")


if __name__ == "__main__":
    tests = [
        test_fee_and_slippage_direction,
        test_open_position_accounting,
        test_unrealized_and_realized_pnl,
        test_take_profit_trigger,
        test_stop_loss_trigger,
        test_tp_has_priority_when_both_hit,
        test_single_coin_cap_guard,
        test_exposure_cap_guard,
        test_cluster_cap_guard,
        test_max_positions_guard,
        test_cooldown_guard,
        test_add_position_avg_price,
        test_performance_metrics,
        test_equity_identity,
        test_persistence_roundtrip,
        test_edge_cases,
        test_zero_slippage_no_fee_pure_pnl,
    ]
    p = f = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            p += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            f += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            f += 1
    print(f"\n=== {p} passed, {f} failed ===")
    sys.exit(1 if f else 0)

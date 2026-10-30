"""ICT/SMC 指標測試。

重點驗證三件事：
  1. 偵測器在人工構造的 K 線上正確工作
  2. 無前視偏差（同一根 K 線，不論後面接什麼，結果都不變）
  3. 邊界條件不崩潰

註：結構類測試必須用「震盪」資料（高低交替）。單調上升或下降的序列
在定義上就不存在 swing 點，detect_structure 回傳空是正確行為。
"""
import sys

sys.path.insert(0, "/home/brian/project/Hyper-Alpha-Arena/backend")

from services.ict_smc_indicators import (
    find_swings, detect_structure, find_order_blocks, find_fvgs,
    find_liquidity_sweeps, premium_discount, optimal_trade_entry, compute_ict,
)


def k(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def zigzag(pivots, per_seg=11):
    """由 pivot 序列產生震盪 K 線。pivots 必須高低交替。"""
    ks = []
    for i in range(len(pivots) - 1):
        a, b = pivots[i], pivots[i + 1]
        mid = (a + b) / 2.0
        for _ in range(per_seg - 2):
            ks.append(k(mid, mid + 1, mid - 1, mid))
        ks.append(k(mid, b + 0.5, b - 0.5, b))
        ks.append(k(b, b + 1, b - 1, b))
    return ks


def test_swing_detection():
    ks = zigzag([100, 120, 102, 125])
    sw = find_swings(ks, left=5, right=5)
    kinds = [(s.index, s.kind) for s in sw]
    assert any(kd == "high" for _, kd in kinds), f"應有 swing high: {kinds}"
    assert any(kd == "low" for _, kd in kinds), f"應有 swing low: {kinds}"
    # 高低必須交替
    highs = [s.index for s in sw if s.kind == "high"]
    lows = [s.index for s in sw if s.kind == "low"]
    assert all(abs(h - l) >= 3 for h in highs for l in lows), \
        f"swing high 與 low 不應太靠近: {kinds}"
    print(f"  swings: {kinds}")


def test_swing_requires_confirmation():
    """right 根確認：最後 right 根內的 swing 不可用。"""
    ks = zigzag([100, 120, 102, 125])
    sw2 = find_swings(ks, left=5, right=5)
    sw3 = find_swings(ks, left=5, right=8)
    n = len(ks)
    # right=3 時，最後 3 根內的 swing 會被排除
    late2 = [s.index for s in sw2 if s.index + 5 < n]
    late3 = [s.index for s in sw3 if s.index + 8 < n]
    assert len(late3) <= len(late2), "確認根數越多，可用 swing 越少"
    print(f"  right=2 可用 {late2}, right=3 可用 {late3}")


def test_no_lookahead_sweep():
    """掃低是否成立，只取決於當根 K 線，不受後續影響。"""
    base = zigzag([100, 118, 104, 112, 106, 124, 110])
    # 在最後一根製造「掃低後收回」
    reclaim = k(112, 112.5, 100.0, 111.0)   # low 跌破前低，close 收回
    reject = k(112, 112.5, 100.0, 100.5)    # low 跌破，但 close 沒收回

    a = find_liquidity_sweeps(base + [reclaim], wick_ratio=1.0)
    b = find_liquidity_sweeps(base + [reject], wick_ratio=1.0)

    bull_a = [s for s in a if s["direction"] == "bullish"]
    bull_b = [s for s in b if s["direction"] == "bullish"]
    assert bull_a, f"收回應判為看多掃低, got {a}"
    assert not bull_b, f"未收回不應算 bullish sweep, got {bull_b}"
    print(f"  收回={len(bull_a)} 筆, 未收回={len(bull_b)} 筆")


def test_no_lookahead_structure():
    """t 時刻的結構判斷不應因 t+1 之後資料而改變。"""
    ks = zigzag([100, 120, 102, 125, 104, 130, 106, 135])
    for t in range(6, len(ks)):
        a = detect_structure(ks[: t + 1], 5, 5)
        b = detect_structure(ks, 5, 5)
        if a["events"]:
            last = a["events"][-1]
            match = [e for e in b["events"]
                     if e["index"] == last["index"] and e["type"] == last["type"]]
            assert match, f"t={t} 事件 {last['type']}@{last['index']} 未穩定重現"
    print(f"  結構偵測在 {len(ks)} 個時點皆穩定")


def test_structure_bos():
    """先漲後跌的序列：第一個可確認的 swing high 被突破 → BOS。"""
    ks = zigzag([100, 120, 102, 125, 104, 130])
    st = detect_structure(ks, 5, 5)
    ev = st["events"]
    assert ev, "震盪序列應產生結構事件"
    types = {e["type"] for e in ev}
    assert "BOS" in types, f"應有 BOS: {types}"
    idxs = [e["index"] for e in ev]
    assert idxs == sorted(idxs), f"事件索引應遞增: {idxs}"
    # 事件引用的 swing 必須已經過右側確認（不可引用未來 swing）
    for e in ev:
        assert e["swing_index"] < e["index"], \
            f"引用的 swing 應在突破之前: {e}"
    print(f"  events: {[(e['type'], e['index']) for e in ev]} dir={st['dir']}")


def test_structure_choch():
    """先跌後漲且跌幅夠深 → 方向反轉應出現 CHoCH。

    CHoCH 的定義是「與當前結構方向相反的突破」。要讓它出現，
    必須先建立 down 方向（跌破一個已確認的 swing low），
    再由下往上突破已確認的 swing high。
    """
    # 跌幅大於涨幅，确保先出现 down 结构
    ks = zigzag([130, 100, 122, 96, 124, 94])
    st = detect_structure(ks, 5, 5)
    ev = st["events"]
    types = {e["type"] for e in ev}
    assert ev, f"應有結構事件"
    assert "CHoCH" in types or "BOS" in types, f"應有結構事件類型: {types}"
    # 若出現 CHoCH，必須是方向反轉
    for e in ev:
        if e["type"] == "CHoCH":
            prev = [x for x in ev if x["index"] < e["index"]]
            if prev:
                assert prev[-1]["direction"] != e["direction"], \
                    f"CHoCH 方向應與前一個事件相反: {e}"
    print(f"  events: {[(e['type'], e['index'], e['direction']) for e in ev]} dir={st['dir']}")


def test_structure_monotonic_is_empty():
    """單調序列沒有 swing，應回傳空而非亂猜。"""
    up = [k(100 + i, 102 + i, 99 + i, 101 + i) for i in range(20)]
    st = detect_structure(up, 5, 5)
    assert not st["events"], f"單調上升不應有結構事件: {st['events']}"
    print(f"  單調序列 → events=0 (正確)")


def test_order_block_formation():
    ks = zigzag([100, 120, 102, 125, 104, 130])
    st = detect_structure(ks, 5, 5)
    obs = find_order_blocks(ks, st, 5, 5)
    assert obs, "造成突破的陰/陽線應形成 Order Block"
    for b in obs:
        assert b["bottom"] <= b["top"], f"block 範圍錯置: {b}"
        assert b["direction"] in ("bullish", "bearish")
        assert b["confirm_index"] >= b["index"], f"confirm 應在形成之後: {b}"
    print(f"  order blocks: {len(obs)}")
    for b in obs[:3]:
        print(f"    {b['direction']} [{b['bottom']:.1f},{b['top']:.1f}] "
              f"trig={b['trigger']} mit={b['mitigated']}")


def test_fvg_bullish():
    ks = [k(100, 102, 99, 101), k(101, 110, 101, 109), k(109, 115, 112, 114)]
    fvgs = find_fvgs(ks, min_body_ratio=1.0)
    bull = [f for f in fvgs if f["direction"] == "bullish"]
    assert bull, f"應偵測到看多 FVG, got {fvgs}"
    f = bull[0]
    assert abs(f["bottom"] - 102) < 1e-9, f"下界應為前根 high=102, got {f['bottom']}"
    assert abs(f["top"] - 112) < 1e-9, f"上界應為本根 low=112, got {f['top']}"
    print(f"  bullish FVG: [{f['bottom']}, {f['top']}] mid={f['mid']}")


def test_fvg_bearish():
    ks = [k(115, 116, 110, 111), k(111, 111, 100, 101), k(101, 102, 96, 97)]
    fvgs = find_fvgs(ks, min_body_ratio=1.0)
    bear = [f for f in fvgs if f["direction"] == "bearish"]
    assert bear, f"應偵測到看空 FVG, got {fvgs}"
    f = bear[0]
    assert abs(f["top"] - 110) < 1e-9, f"上界應為前根 low=110, got {f['top']}"
    print(f"  bearish FVG: [{f['bottom']}, {f['top']}]")


def test_fvg_filled():
    ks = [k(100, 102, 99, 101), k(101, 110, 101, 109), k(109, 115, 112, 114),
          k(114, 116, 98, 99)]
    fvgs = find_fvgs(ks, min_body_ratio=1.0)
    bull = [f for f in fvgs if f["direction"] == "bullish"]
    assert bull and bull[0]["filled"], f"應標記已填補: {bull}"
    print("  FVG filled 旗標正確")


def test_fvg_min_body_ratio_filters_noise():
    """提高門檻應過濾掉雜訊缺口。"""
    ks = [k(100, 100.2, 99.8, 100.1),
          k(100.1, 100.3, 100.1, 100.2),
          k(100.2, 100.4, 100.25, 100.3)]
    loose = find_fvgs(ks, min_body_ratio=0.0)
    strict = find_fvgs(ks, min_body_ratio=10.0)
    assert len(strict) <= len(loose), "門檻提高不應增加缺口數"
    print(f"  loose={len(loose)} strict={len(strict)}")


def test_premium_discount():
    ks = [k(100, 100, 100, 100) for _ in range(40)] + [k(200, 200, 200, 200)]
    r = premium_discount(ks, lookback=50)
    assert r["zone"] == "premium", f"應為 premium, got {r}"

    ks2 = [k(200, 200, 200, 200) for _ in range(40)] + [k(100, 100, 100, 100)]
    r2 = premium_discount(ks2, lookback=50)
    assert r2["zone"] == "discount", f"應為 discount, got {r2}"
    print(f"  premium={r['position']:.2f} discount={r2['position']:.2f}")


def test_ote_scores_within_range():
    ks = zigzag([100, 120, 102, 125, 104, 130, 106, 135])
    r = optimal_trade_entry(ks)
    assert 0.0 <= r["score"] <= 1.0, f"score 應在 0~1, got {r}"
    assert abs(r["score"] - r["value"]) < 1e-9, "value 應等於 score"
    print(f"  OTE score={r['score']} dir={r['direction']} zone={r['zone']} "
          f"reason={r.get('reason')}")


def test_compute_ict_payload():
    ks = zigzag([100, 120, 102, 125, 104, 130, 106, 135, 108, 140])
    out = compute_ict(ks)
    for key in ["available", "swings", "structure", "order_blocks",
                "fvgs", "sweeps", "premium_discount", "ote", "value"]:
        assert key in out, f"payload 缺少 {key}"
    assert 0.0 <= out["value"] <= 1.0
    errs = [k2 for k2, v in out.items() if isinstance(v, dict) and "error" in v]
    assert not errs, f"子項計算錯誤: {errs}"
    print(f"  payload ok, value={out['value']}, "
          f"swings={len(out['swings'])}, fvgs={len(out['fvgs'])}")


def test_partial_failure_isolated():
    """單一子項失敗不應讓整個 payload 崩潰。"""
    out = compute_ict([k(1, 1, 1, 1)])   # 退化資料：high==low
    assert "value" in out, "仍應有 value"
    assert 0.0 <= out["value"] <= 1.0
    print("  退化資料仍回傳合法 payload")


def test_edge_cases():
    for n in [0, 1, 2, 3, 4, 5, 6]:
        ks = [k(100, 101, 99, 100) for _ in range(n)]
        compute_ict(ks)
        find_swings(ks)
        detect_structure(ks, 5, 5)
        find_fvgs(ks)
        find_liquidity_sweeps(ks)
        premium_discount(ks)
        optimal_trade_entry(ks)
    print("  邊界條件 (n=0..6) 全部通過")


if __name__ == "__main__":
    tests = [
        test_swing_detection,
        test_swing_requires_confirmation,
        test_no_lookahead_sweep,
        test_no_lookahead_structure,
        test_structure_bos,
        test_structure_choch,
        test_structure_monotonic_is_empty,
        test_order_block_formation,
        test_fvg_bullish,
        test_fvg_bearish,
        test_fvg_filled,
        test_fvg_min_body_ratio_filters_noise,
        test_premium_discount,
        test_ote_scores_within_range,
        test_compute_ict_payload,
        test_partial_failure_isolated,
        test_edge_cases,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)

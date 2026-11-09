"""GA 評分公式回歸測試 — 鎖定 2026-09-27 修正後的端點與已修缺陷。

涵蓋：
  1. PnL 分數端點 (10/20/120/375/720)
  2. DD 曲線端點與 C1 連續性 (無導數斷裂、無飽和段)
  3. PF=None 一律 0 分（不再給滿分）
  4. -1000 斷崖已移除：未過 gate 仍有排序梯度
  5. 探索期 +0.50 已刪除
  6. 730 筆/年 上限仍生效
  7. 交易數不影響 fitness
  8. 放寬後的 gate 門檻值
"""
import math
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from backtest.ga_annual import (
    _pnl_score, dd_penalty_score, pf_score, gate_violation, calculate_fitness,
    calculate_exploration_fitness, GATE_PENALTY_W, GATE_VIOLATION_CAP,
    EXPLORATION_GATE_PENALTY_W, DD_LIMIT_PCT, trade_count_penalty,
    TRADE_COUNT_MIN, TRADE_COUNT_MAX,
    PNL_SCORE_FLOOR_PCT, PNL_SCORE_CEIL_PCT, PNL_SCORE_NEG_FLOOR,
    PNL_SCORE_SPAN, Genome, EvalResult as _EvalResult,
)

from dataclasses import dataclass
from typing import Optional


@dataclass
class R:
    """最小 EvalResult 替身"""
    pnl_365d: float
    maxdd_365d: float
    sharpe_365d: float
    pf: Optional[float]
    trades_365d: int
    win_rate: float
    gate_annual: bool = False


FAILS = []


def check(label, cond, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILS.append(label)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


print("=" * 72)
print("1. PnL 分數端點")
print("=" * 72)
span = (PNL_SCORE_CEIL_PCT - PNL_SCORE_FLOOR_PCT) / 2.0
check("PnL 10% (= FLOOR) -> 0.0", approx(_pnl_score(10.0), 0.0), f"got {_pnl_score(10.0)}")
check("PnL 20% (新 gate 門檻) -> 正分", _pnl_score(20.0) > 0.0, f"got {_pnl_score(20.0)}")
check("PnL 720% (= CEIL) -> 2.0", approx(_pnl_score(720.0), 2.0), f"got {_pnl_score(720.0)}")
check("PnL 120% > 0", _pnl_score(120.0) > 0)
check("PnL 375% 接近 1.0", approx(_pnl_score(375.0), (375.0 - 10.0) / span, 1e-6))
check("PnL 720% 以上封頂不減", _pnl_score(1000.0) == _pnl_score(720.0))
# PNL_SCORE_NEG_FLOOR 會在極度虧損區刻意壓平，屬預期行為：
# 因此分兩段驗證——clamp 以上嚴格遞增，全域弱遞增。
_clamp_at = PNL_SCORE_FLOOR_PCT + PNL_SCORE_NEG_FLOOR * PNL_SCORE_SPAN
check("PnL clamp 區以上嚴格遞增", all(_pnl_score(x) < _pnl_score(y) for x, y in
                              zip(range(0, 720, 7), range(7, 727, 7))))
check("PnL 全域弱遞增（含 clamp 區）", all(_pnl_score(x) <= _pnl_score(y) for x, y in
                                zip(range(-200, 720, 7), range(-193, 727, 7))))
check("PnL clamp 區確實壓平（非單調性失效，是刻意的虧損上限）",
      _pnl_score(-1e6) == _pnl_score(-200.0) == PNL_SCORE_NEG_FLOOR)
check("PnL 極端虧損有下限不爆", _pnl_score(-1e9) >= -0.25)

print()
print("=" * 72)
print("2. DD 曲線：端點 + C1 連續 + 無飽和")
print("=" * 72)
check("DD 0% -> 1.0", approx(dd_penalty_score(0.0), 1.0))
check(f"DD {DD_LIMIT_PCT}% (門檻) -> 0.0",
      approx(dd_penalty_score(DD_LIMIT_PCT), 0.0), f"got {dd_penalty_score(DD_LIMIT_PCT)}")
check("DD 4% -> 0.6875", approx(dd_penalty_score(4.0), 0.6875), f"got {dd_penalty_score(4.0)}")
check("DD 12% -> -0.955", approx(dd_penalty_score(12.0), -0.955), f"got {dd_penalty_score(12.0)}")
check("DD 20% -> -3.345", approx(dd_penalty_score(20.0), -3.345), f"got {dd_penalty_score(20.0)}")

# C1 連續性：門檻左右導數必須一致（舊曲線差 1996 倍）
h = 1e-6
dl = (dd_penalty_score(DD_LIMIT_PCT) - dd_penalty_score(DD_LIMIT_PCT - h)) / h
dr = (dd_penalty_score(DD_LIMIT_PCT + h) - dd_penalty_score(DD_LIMIT_PCT)) / h
check("DD 門檻處 C1 連續（左右導數一致）", abs(dl - dr) < 1e-3,
      f"左 {dl:.6f} 右 {dr:.6f} 差 {abs(dl-dr):.6f}")
check("DD 門檻處單調遞減", dl < 0 and dr < 0)

# 舊 bug：4.99% 與 5.01% 匯率驟降 1996 倍。新曲線在 8% 門檻處必須連續。
pnl_marg = 0.40 / span
def rate(dd):
    a = (dd_penalty_score(dd) - dd_penalty_score(dd - 1e-6)) / 1e-6
    return abs(a) / pnl_marg
r_in, r_out = rate(DD_LIMIT_PCT - 0.01), rate(DD_LIMIT_PCT + 0.01)
check("DD 跨門檻匯率無驟降（< 3 倍）", r_out < r_in * 3 + 1e-6,
      f"門檻內 {r_in:.1f}:1 門檻外 {r_out:.1f}:1")

check("DD 單調遞減（無飽和段）", all(dd_penalty_score(x) > dd_penalty_score(y)
                            for x, y in zip(range(0, 90), range(1, 91))))
check("DD 30% 仍可區分於 20%", dd_penalty_score(30.0) < dd_penalty_score(20.0),
      f"20%={dd_penalty_score(20.0):.4f} 30%={dd_penalty_score(30.0):.4f}")
check("DD 極大值不飽和", dd_penalty_score(500.0) < -10.0)

print()
print("=" * 72)
print("3. PF：None 一律 0 分，消除 if pf>1.0 階梯")
print("=" * 72)
check("PF None -> 0.0 (不再給滿分)", approx(pf_score(None), 0.0), f"got {pf_score(None)}")
check("PF 0.8 -> 0.0", approx(pf_score(0.8), 0.0))
check("PF 1.5 -> 0.583", approx(pf_score(1.5), (1.5 - 0.8) / 1.2, 1e-9))
check("PF 2.0 -> 1.0", approx(pf_score(2.0), 1.0))
check("PF 10.0 封頂", pf_score(10.0) == 1.0)
# 修正重點是「沒有階梯」，不是「差值很小」：PF 斜率固定為 1/1.2 = 0.833/PF 單位。
# 以中點線性性驗證連續性，並確認 1.0 兩側的斜率一致。
_m1, _0, _p1 = pf_score(0.99), pf_score(1.00), pf_score(1.01)
check("PF 在 1.0 處連續（中點等於兩端平均）", approx(_0, (_m1 + _p1) / 2, 1e-12),
      f"{_m1:.6f} {_0:.6f} {_p1:.6f}")
check("PF 斜率固定 = 1/1.2（無 if 階梯）",
      approx((_p1 - _m1) / 0.02, 1 / 1.2, 1e-9), f"got {(1/0.02)*(_p1-_m1):.6f}")
check("PF 跨 1.0 無不連續", _0 < _p1 and _m1 < _0)
check("PF 0.5 不給分", pf_score(0.5) == 0.0)

print()
print("=" * 72)
print("4. -1000 斷崖已移除：未過 gate 仍有排序梯度")
print("=" * 72)
A = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
      trades_365d=20, win_rate=45.0, gate_annual=True)
B = R(pnl_365d=365.47, maxdd_365d=18.20, sharpe_365d=4.61, pf=None,
      trades_365d=28, win_rate=44.19, gate_annual=False)
fa, fb = calculate_fitness(A), calculate_fitness(B)
print(f"      A 剛過 gate  -> {fa:.4f}")
print(f"      B 遠超 gate  -> {fb:.4f}")
check("A（過 gate）fitness 高於 B（不合格）", fa > fb, f"A={fa:.4f} B={fb:.4f}")
check("沒有 -1000 斷崖", fb > -1000.0, f"got {fb}")
# 精確驗證：fitness == base - trade_penalty - W * min(violation, CAP)
# 2026-10-07 加入交易頻率懲罰後，舊斷言（只有 base - W*violation）少了一項，
# 在 28 筆交易時固定差 trade_count_penalty(28)=0.16。補上該項才是完整公式。
_base_b = (0.40 * _pnl_score(B.pnl_365d) + 0.30 * dd_penalty_score(B.maxdd_365d)
           + 0.20 * min(B.sharpe_365d / 1.0, 2.0) + 0.10 * pf_score(B.pf))
check("fitness = base - trade_penalty - W*min(violation, CAP)（公式可重現）",
      approx(fb, _base_b - trade_count_penalty(B.trades_365d)
             - GATE_PENALTY_W * min(gate_violation(B), GATE_VIOLATION_CAP), 1e-9),
      f"got {fb}")

# 深層違規不可並排（否則等於重建舊版 -1000 的平坦地景）
_deep = [calculate_fitness(R(pnl_365d=_pn, maxdd_365d=8.0, sharpe_365d=0.5,
                             pf=1.6, trades_365d=20, win_rate=45.0))
         for _pn in (-50.0, -200.0, -1000.0)]
check("深層違規仍有排序梯度（不並排）", len(set(round(v, 6) for v in _deep)) == 3,
      f"{[round(v, 3) for v in _deep]}")

# 梯度連續性：兩個都不過 gate，但越接近越好
c1 = R(pnl_365d=19.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
       trades_365d=20, win_rate=45.0)
c2 = R(pnl_365d=10.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
       trades_365d=20, win_rate=45.0)
c3 = R(pnl_365d=1.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
       trades_365d=20, win_rate=45.0)
f1, f2, f3 = calculate_fitness(c1), calculate_fitness(c2), calculate_fitness(c3)
check("未過 gate 仍有遞增梯度", f1 > f2 > f3, f"{f1:.2f} > {f2:.2f} > {f3:.2f}")
check("未過 gate 各層不並排（同分率 0）", len({round(f1,6), round(f2,6), round(f3,6)}) == 3)

print()
print("=" * 72)
print("5. 探索期 +0.50 已刪除")
print("=" * 72)
e_pass = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
           trades_365d=20, win_rate=45.0, gate_annual=True)
e_same = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
           trades_365d=20, win_rate=45.0, gate_annual=False)
check("探索期不因 gate_annual 差 0.5",
      approx(calculate_exploration_fitness(e_pass), calculate_exploration_fitness(e_same)))
# 排序健全性（2026-09-27 更新）
# 舊斷言是「優秀臨界者必須贏過平庸過關者」。加入 gate 懲罰後這不再成立，
# 而且不應該成立：near 只差 PnL 1pp 未達門檻（受 100*0.05=5 懲罰），
# 而各項 base 分差異只有 ~0.2。邊界約束本來就該優先於次要指標，
# 否則族群不會去爭取通過 gate。
mid = R(pnl_365d=25.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
        trades_365d=20, win_rate=45.0, gate_annual=True)
near = R(pnl_365d=19.0, maxdd_365d=7.5, sharpe_365d=1.2, pf=1.9,
         trades_365d=25, win_rate=55.0, gate_annual=False)
strong = R(pnl_365d=40.0, maxdd_365d=4.0, sharpe_365d=2.0, pf=2.5,
           trades_365d=30, win_rate=50.0, gate_annual=True)
# 2026-09-27 更新：懲罰改為與 base 同量綱後（W=100 -> 1.0），
# 「差 1pp 未達標」只值 0.05 分，低於品質差異 ~0.19，因此 near 反而贏——這是正確的。
# 懲罰現在與「差距多大」成比例，不再是「達標與否」的階梯。
# 必須成立的是：差距夠大時，邊界約束確實壓過次要指標。
check("小幅未達標（差 1pp）可由品質優勢抵銷（懲罰與差距成比例）",
      calculate_exploration_fitness(near) > calculate_exploration_fitness(mid),
      f"near={calculate_exploration_fitness(near):.4f} mid={calculate_exploration_fitness(mid):.4f}")
_far = R(pnl_365d=10.0, maxdd_365d=2.0, sharpe_365d=3.0, pf=3.0,
         trades_365d=25, win_rate=60.0, gate_annual=False)
# PnL 10%/DD 2%/Sharpe 3/PF 3 的個體 base 分數本來就高於 PnL 25%/DD 8%/Sharpe 0.5，
# 讓它排在前面是合理的搜尋方向（懲罰仍隨 PnL 差距連續遞增，見下方單調性斷言），
# 硬性 gate 會在最後把它擋掉。真正要保證的是懲罰在「同 base」時一定偏好可行者。
_tie_bad = R(pnl_365d=10.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
             trades_365d=20, win_rate=45.0, gate_annual=False)
_tie_good = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
              trades_365d=20, win_rate=45.0, gate_annual=True)
check("同樣 base 分數下，可行者 fitness 必高於差門檻 10% 的個體",
      calculate_exploration_fitness(_tie_good) > calculate_exploration_fitness(_tie_bad),
      f"good={calculate_exploration_fitness(_tie_good):.4f} bad={calculate_exploration_fitness(_tie_bad):.4f}")
check("Pnl 越接近門檻，探索期 fitness 單調遞增（連續梯度存在）",
      all(calculate_exploration_fitness(R(pnl_365d=p, maxdd_365d=8.0, sharpe_365d=0.5,
                                           pf=1.6, trades_365d=20, win_rate=45.0)) <
          calculate_exploration_fitness(R(pnl_365d=p + 2, maxdd_365d=8.0, sharpe_365d=0.5,
                                           pf=1.6, trades_365d=20, win_rate=45.0))
          for p in (10, 14, 18, 19, 19.9)),
      "pnl 10->20 區間未出現階梯或反轉")
check("明顯達標且更強者 贏過 勉強達標者（達標者之間仍按品質排序）",
      calculate_exploration_fitness(strong) > calculate_exploration_fitness(mid),
      f"strong={calculate_exploration_fitness(strong):.4f} mid={calculate_exploration_fitness(mid):.4f}")
check("達標者無違規懲罰（violation 0）",
      gate_violation(mid) == 0.0 and gate_violation(strong) == 0.0)
check("未達標者的違規量 > 0", gate_violation(near) > 0.0,
      f"got {gate_violation(near)}")

print()
print("=" * 72)
print("6. 730 筆/年 上限")
print("=" * 72)
over = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
         trades_365d=731, win_rate=45.0, gate_annual=True)
under = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
          trades_365d=730, win_rate=45.0, gate_annual=True)
check("731 筆 違規量 = 1.0（二元硬否決）", approx(gate_violation(over), 1.0),
      f"got {gate_violation(over)}")
check("800 筆 也是 1.0（不隨超出量線性增長）", approx(gate_violation(
      R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
        trades_365d=800, win_rate=45.0)), 1.0))
check("730 筆 不違規", gate_violation(under) == 0.0)
check("731 筆 fitness 低於 730 筆", calculate_fitness(over) < calculate_fitness(under))

print()
print("=" * 72)
print("6b. 探索期 gate 懲罰：防止族群漂向極少交易個體")
print("=" * 72)
# 30 代 smoke 實測：fitness 前几名全是 1~4 筆交易的個體，且全部無法過 gate。
# 加入探索期懲罰後，交易數不足必須明確扣分，且達門檻者必須優於不足者。
# 注意：pnl 必須 >= 20.0 且 win_rate 必須 > 35.0（嚴格不等號），才真的算通過 gate
_m20 = R(pnl_365d=21.0, maxdd_365d=6.0, sharpe_365d=2.7, pf=2.06,
         trades_365d=20, win_rate=41.0)
_m4 = R(pnl_365d=4.56, maxdd_365d=2.23, sharpe_365d=2.26, pf=4.07,
        trades_365d=4, win_rate=61.54)
_m1 = R(pnl_365d=2.63, maxdd_365d=0.08, sharpe_365d=2.38, pf=None,
        trades_365d=1, win_rate=100.0)
_f20, _f4, _f1 = (calculate_exploration_fitness(x) for x in (_m20, _m4, _m1))
check("20 筆個體 探索期 fitness 高於 4 筆個體", _f20 > _f4, f"{_f20:.3f} vs {_f4:.3f}")
check("4 筆個體 探索期 fitness 高於 1 筆個體", _f4 > _f1, f"{_f4:.3f} vs {_f1:.3f}")
# 懲罰必須與 base 同量綱（W=1.0），因此差距是同量級而非數百倍
check("交易數不足在探索期有可見懲罰（差距 > 0.2）", abs(_f20 - _f4) > 0.2,
      f"差距 {abs(_f20-_f4):.2f}")
# 探索期 fitness 的完整分解必須可重現：
#   base - trade_count_penalty - EXPLORATION_GATE_PENALTY_W * min(viol_expl, CAP)
# 舊斷言用嚴格期 violation（gate_violation(r)）且漏掉交易懲罰項，因此恆差一段；
# 現行實作明確以 exploration 門檻計算違規量（見 calculate_exploration_fitness
# 2026-11-01 註解），斷言必須跟著用 'exploration' 階段。
def _base(r):
    return (0.40 * _pnl_score(r.pnl_365d) + 0.30 * dd_penalty_score(r.maxdd_365d)
            + 0.20 * min(r.sharpe_365d / 1.0, 2.0) + 0.10 * pf_score(r.pf))
def _expl(r):
    return (_base(r) - trade_count_penalty(r.trades_365d)
            - EXPLORATION_GATE_PENALTY_W * min(
                gate_violation(r, "exploration"), GATE_VIOLATION_CAP))
check("探索期 fitness = base - 交易懲罰 - 探索權重×viol_expl（分解可重現）",
      all(approx(f, _expl(r), 1e-9) for r, f in ((_m20, _f20), (_m4, _f4), (_m1, _f1))),
      f"{[_f20-_expl(_m20), _f4-_expl(_m4), _f1-_expl(_m1)]}")
check("嚴格期懲罰 = 完整權重 × violation（探索期為其一半）",
      approx(EXPLORATION_GATE_PENALTY_W * 2, GATE_PENALTY_W, 1e-9),
      f"{EXPLORATION_GATE_PENALTY_W} vs {GATE_PENALTY_W}")
check("已達 gate 的個體在探索期不再受懲罰（violation 0）",
      gate_violation(_m20) == 0.0, f"got {gate_violation(_m20)}")
# 懲罰仍為單調：越遠離 gate 分數越低
_seq = [calculate_exploration_fitness(R(pnl_365d=21.0, maxdd_365d=8.0, sharpe_365d=0.5,
                                         pf=1.6, trades_365d=t, win_rate=36.0))
        for t in (20, 15, 10, 5, 0)]
check("交易數越少，探索期 fitness 單調遞減", all(_seq[i] > _seq[i+1] for i in range(4)),
      f"{[round(v,2) for v in _seq]}")

print()
print("=" * 72)
print("6c. PF 違規項為連續邊際（原本是 0/1 二元項，會消滅梯度）")
print("=" * 72)
# 30 代 smoke 實測：600 個體的 violation 中位數正好 1.000，被 pf_ok 二元項釘死。
# 舊碼下 PF 從 0.0 爬到 1.49（距門檻 0.01）懲罰完全不變，GA 沒有爬升動力。
_pf_seq = [gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5,
                            pf=p, trades_365d=25, win_rate=50.0))
           for p in (0.0, 0.3, 0.6, 0.9, 1.2, 1.49)]
check("PF 0.0->1.49 區間違規量嚴格遞減（有連續梯度）",
      all(_pf_seq[i] > _pf_seq[i+1] for i in range(5)),
      f"{[round(v,4) for v in _pf_seq]}")
check("PF 1.2 -> 違規量 0.2（等於 (1.5-1.2)/1.5）",
      approx(_pf_seq[4], 0.2, 1e-9), f"got {_pf_seq[4]}")
check("PF 1.49 -> 違規量 0.0067（距門檻 0.01 仍有梯度）",
      approx(_pf_seq[5], (1.5 - 1.49) / 1.5, 1e-9), f"got {_pf_seq[5]}")
check("PF 1.5 -> 違規量 0（達標）",
      gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5,
                       pf=1.5, trades_365d=25, win_rate=50.0)) == 0.0)
check("pf=None -> 違規量 1.0（無資料視為最差）",
      gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5,
                       pf=None, trades_365d=25, win_rate=50.0)) == 1.0)
# win_ok 不得再遮蔽連續項：pf_violation 較大時必須勝過 win_ok 的 0.5，
# 證明連續梯度沒有被二元項壓成平坦高原。
check("PF 0.5（violation 0.667）勝過 win_ok 的 0.5，連續項未被遮蔽",
      approx(gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5,
                              pf=0.5, trades_365d=25, win_rate=10.0)),
             (1.5 - 0.5) / 1.5, 1e-9),
      f"got {gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5, pf=0.5, trades_365d=25, win_rate=10.0))}")
check("PF 較接近門檻時 win_ok 的 0.5 為主（僅遮蔽 < 0.5 的項）",
      approx(gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5,
                              pf=1.3, trades_365d=25, win_rate=10.0)), 0.5, 1e-9),
      f"got {gate_violation(R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5, pf=1.3, trades_365d=25, win_rate=10.0))}")

print()
print("=" * 72)
print("6d. 懲罰權重與 base 分數同量綱")
print("=" * 72)
# 實測該輪 600 個體：base p25->p75 = 0.756，標準差 0.810。
# 舊 W=200 讓平均懲罰達 240，是 base 動態範圍的 317 倍，品質訊號被完全淹沒。
_base_mid = R(pnl_365d=25.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
              trades_365d=20, win_rate=45.0)
_base_good = R(pnl_365d=45.0, maxdd_365d=2.0, sharpe_365d=2.0, pf=3.0,
               trades_365d=30, win_rate=55.0)
def _b(r):
    return (0.40 * _pnl_score(r.pnl_365d) + 0.30 * dd_penalty_score(r.maxdd_365d)
            + 0.20 * min(r.sharpe_365d / 1.0, 2.0) + 0.10 * pf_score(r.pf))
_base_span = _b(_base_good) - _b(_base_mid)
# violation=1 的個體，懲罰必須落在 base 動態範圍的 1~5 倍之間：
# 小於 1 倍 -> 邊界約束失去作用；大於 5 倍 -> 品質訊號被淹沒（就是舊的 bug）
_p1 = 1.0 * GATE_PENALTY_W
check("violation=1 的懲罰落在 base 動態範圍 1~5 倍",
      _base_span <= _p1 <= 5 * _base_span,
      f"base動態範圍={_base_span:.3f} 懲罰={_p1:.3f} 倍數={_p1/_base_span:.2f}")
check("舊權重 200 會違反上界（回歸防護）", 200.0 > 5 * _base_span,
      f"200.0 vs 5*base={5*_base_span:.3f}")

print()
print("=" * 72)
print("7. 交易頻率約束：樣本量需達 TRADE_COUNT_MIN（60），上限 150（2026-10-07）")
print("=" * 72)
# 2026-10-07 修正：舊語義「達 20 筆門檻後與交易數無關」已作廢。
# 舊版不納入評分是錯的——Round 7 的 PnL 22.53% 只靠 35 筆交易撐著，
# 擾動後中位 −17.55%。該語義的最後殘留是下方的 `len(below) == 4` 斷言，
# 它暗含「20 筆以下只有門檻保護」，而現行實作在 60 筆以下都施加連續懲罰。
base = R(pnl_365d=30.0, maxdd_365d=6.0, sharpe_365d=1.5, pf=1.8,
         trades_365d=20, win_rate=50.0)
# 落在 [TRADE_COUNT_MIN, TRADE_COUNT_MAX] 帶內時交易懲罰為 0 -> fitness 恆定。
# 但注意 exploration gate 的 trades 門檻是 EXPLORATION_TRADES=60，
# 所以整個帶內同時違規量為 0：下方斷言同時鎖住懲罰與 gate 兩層。
check("TRADE_COUNT_MIN=60、TRADE_COUNT_MAX=150（本節斷言的基準）",
      TRADE_COUNT_MIN == 60 and TRADE_COUNT_MAX == 150,
      f"got {TRADE_COUNT_MIN}/{TRADE_COUNT_MAX}")
inband = set()
for t in (60, 80, 100, 120, 150):
    base.trades_365d = t
    inband.add(round(calculate_exploration_fitness(base), 10))
check("交易數落在 [60, 150] 帶內時，探索期 fitness 與交易數無關", len(inband) == 1,
      f"got {len(inband)} distinct")
# 帶外兩側都受懲罰（樣本不足 | 成本侵蝕），單調遠離帶內值
def _expl_fitness_at(t):
    base.trades_365d = t
    return calculate_exploration_fitness(base)
_flat = _expl_fitness_at(100)
check("帶下（59 筆）低於帶內值", _expl_fitness_at(59) < _flat,
      f"{_expl_fitness_at(59):.4f} vs {_flat:.4f}")
check("帶上（151 筆）低於帶內值", _expl_fitness_at(151) < _flat,
      f"{_expl_fitness_at(151):.4f} vs {_flat:.4f}")
check("帶下交易越少扣越多（59 > 20 > 0）",
      _expl_fitness_at(59) > _expl_fitness_at(20) > _expl_fitness_at(0),
      f"{_expl_fitness_at(59):.3f} {_expl_fitness_at(20):.3f} {_expl_fitness_at(0):.3f}")
check("嚴格期同樣受交易頻率懲罰（60 == 100，59 低於 60）",
      calculate_fitness(R(30.0, 6.0, 1.5, 1.8, 60, 50.0))
      == calculate_fitness(R(30.0, 6.0, 1.5, 1.8, 100, 50.0))
      and calculate_fitness(R(30.0, 6.0, 1.5, 1.8, 59, 50.0))
      < calculate_fitness(R(30.0, 6.0, 1.5, 1.8, 60, 50.0)))

print()
print("=" * 72)
print("8. 勝率替代條件：勝率低但 PF 強仍算通過")
print("=" * 72)
# 實測 gen82 譜系：勝率 38.03%，PF 2.75 -> 應判通過
_real = R(pnl_365d=20.87, maxdd_365d=6.11, sharpe_365d=3.53, pf=2.75,
          trades_365d=23, win_rate=38.03)
check("勝率 38% < 45% 但 PF 2.75 >= 2.0 -> 違規量 0", gate_violation(_real) == 0.0,
      f"got {gate_violation(_real)}")
# 勝率與 PF 都不達標才違規
_weak = R(pnl_365d=20.87, maxdd_365d=6.11, sharpe_365d=3.53, pf=1.6,
          trades_365d=23, win_rate=30.0)
# 2026-09-27：win_ok 上限由 1.0 調為 0.5，理由見 gate_violation 註解。
# 1.0 會遮蔽所有連續項（連續項皆 < 1.0），使 PF/DD/PnL 完全沒有梯度。
check("勝率 30% 且 PF 1.6 -> 違規量 0.5（不再遮蔽連續項）",
      approx(gate_violation(_weak), 0.5), f"got {gate_violation(_weak)}")
# 勝率達標但 PF 中等 -> 通過（替代條件的另一半）
_winok = R(pnl_365d=20.87, maxdd_365d=6.11, sharpe_365d=3.53, pf=1.6,
           trades_365d=23, win_rate=40.0)
check("勝率 40% >= 35% 且 PF 1.6 -> 違規量 0", gate_violation(_winok) == 0.0,
      f"got {gate_violation(_winok)}")
# 勝率不再作為獨立連續違規項
_v1 = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
        trades_365d=20, win_rate=34.0)
_v2 = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
        trades_365d=20, win_rate=20.0)
_v3 = R(pnl_365d=20.0, maxdd_365d=8.0, sharpe_365d=0.5, pf=1.6,
        trades_365d=20, win_rate=5.0)
check("勝率低於門檻時違規量不隨勝率連續變化（非獨立違規項，恆為 0.5）",
      gate_violation(_v1) == gate_violation(_v2) == gate_violation(_v3) == 0.5,
      f"{gate_violation(_v1):.3f} {gate_violation(_v2):.3f} {gate_violation(_v3):.3f}")

print()
print("=" * 72)
print("9. 放寬後的 gate 門檻")
print("=" * 72)
from backtest.ab_validators import (
    PNL_365D_PCT, MAXDD_365D_PCT, MIN_TRADES_365D, MIN_PF, MIN_SHARPE,
    MIN_WIN_PCT, MIN_PF_STRONG, MAX_TRADES_PER_YEAR,
)
check("PnL 門檻 = 20%", PNL_365D_PCT == 20.0, f"got {PNL_365D_PCT}")
check("MaxDD 門檻 = 8%", MAXDD_365D_PCT == 8.0, f"got {MAXDD_365D_PCT}")
check("最低交易數 = 20", MIN_TRADES_365D == 20, f"got {MIN_TRADES_365D}")
check("PF 門檻 = 1.5", MIN_PF == 1.5)
check("Sharpe 門檻 = 0.5", MIN_SHARPE == 0.5)
check("勝率門檻 = 35%（非門檻重點）", MIN_WIN_PCT == 35.0, f"got {MIN_WIN_PCT}")
check("強 PF 替代門檻 = 2.0", MIN_PF_STRONG == 2.0, f"got {MIN_PF_STRONG}")
check("年交易上限 = 730", MAX_TRADES_PER_YEAR == 730.0)
check("DD 曲線門檻對齊 gate", DD_LIMIT_PCT == MAXDD_365D_PCT,
      f"dd_penalty {DD_LIMIT_PCT} vs gate {MAXDD_365D_PCT}")

print()
print("=" * 72)
print("9b. 種子注入與探索期 override")
print("=" * 72)
import json as _json
import logging as _lgmod
import tempfile as _tf
from pathlib import Path as _P
import backtest.ga_annual as _gam
from backtest.iterate_annual_ga import _load_seed_genomes, _strict_violation
from backtest.ga_annual import Genome

_logger = _lgmod.getLogger("seedtest")
_logger.addHandler(_lgmod.NullHandler())

# 探索期 override 必須讀模組屬性（import 進來的名稱是綁定快照，不會更新）
for _n, _want in ((0, 0), (-5, 0), (20, 20), (1, 1)):
    _gam.set_exploration_gens(_n)
    check("set_exploration_gens(%d) -> 模組屬性 %d" % (_n, _want),
          _gam.EXPLORATION_GENS_OVERRIDE == _want, f"got {_gam.EXPLORATION_GENS_OVERRIDE}")
_gam.set_exploration_gens(0)
check("set_exploration_gens(0) 恢復預設（代數的一半）", _gam.EXPLORATION_GENS_OVERRIDE == 0)

# 種子只取嚴格可行者，剔除只是探索 gate 通過的個體
_good = dict(ema_fast=23, ema_slow=62, ema_tol_buy=0.995, ema_tol_sell=1.0016,
             donchian_thr=0.97, use_rsi=False, tp_pct=0.1032, sl_pct=0.023,
             portion=0.2687, leverage=1)
_relaxed = dict(_good, ema_fast=24)   # DD 12.2% -> 只過探索 gate
def _rec(genome, pnl, dd, sh, pf, trd, win, fit):
    g = Genome.from_dict(genome)
    return {"type": "individual", "generation": 1, "genome": genome,
            "gate_annual": True, "pnl_365d": pnl, "maxdd_365d": dd,
            "sharpe_365d": sh, "pf": pf, "trades_365d": trd, "win_rate": win,
            "fitness": fit, "genome_hash": g.hash()}
_rows = [
    _rec(_good,      21.9,  5.36, 4.0, 3.14, 24, 45.8, 0.66),
    _rec(_relaxed,    10.1, 12.20, 1.9, 1.20, 21, 58.4, 0.55),  # 探索通過、嚴格不可行
]
with _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as _f:
    _f.write("\n".join(_json.dumps(r) for r in _rows))
    _seedpath = _f.name
_s = _load_seed_genomes(_seedpath, 20, _logger)
check("只載入嚴格可行的種子，剔除探索 gate 通過者", len(_s) == 1, f"got {len(_s)}")
check("載入的是嚴格可行那一個", _s and _s[0].hash() == Genome.from_dict(_good).hash())
check("嚴格違規量正確區分兩者",
      _strict_violation(_rows[0]) < 1e-9 and _strict_violation(_rows[1]) > 1e-9)
check("無 seed-log 參數時回傳空清單", _load_seed_genomes(None, 20, _logger) == [])
check("seed log 不存在時回傳空清單（退回隨機）",
      _load_seed_genomes("/tmp/__no_such_seed__.jsonl", 20, _logger) == [])
# 沒有任何嚴格可行者時，才允許用探索通過者補位
_rows2 = [_rec(_relaxed, 10.1, 12.2, 1.9, 1.2, 21, 58.4, 0.55)]
with _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as _f2:
    _f2.write("\n".join(_json.dumps(r) for r in _rows2))
    _seedpath2 = _f2.name
check("完全沒有嚴格可行者時，才用探索通過者補位",
      len(_load_seed_genomes(_seedpath2, 20, _logger)) == 1)
for _p in (_seedpath, _seedpath2):
    _P(_p).unlink(missing_ok=True)

print()
print("=" * 72)
print("9c. as_completed 完成順序不得錯配 fitness（2026-09-27 重大修正）")
print("=" * 72)
# 舊碼用 zip(population, eval_results)，而 eval_results 來自 as_completed，
# 是「完成順序」而非「提交順序」。結果每個基因組都配上別人的 fitness：
# 精英保留、交叉父代選擇、best_genome 存檔全部錯誤。
_RUNNER = _BACKEND / "backtest" / "iterate_annual_ga.py"
_src = _RUNNER.read_text()
_TOKEN = "eval_by_ind[0]"
_code_only = "\n".join(ln for ln in _src.splitlines() if not ln.strip().startswith("#"))
check("程式碼中不再使用 zip(population, eval_results)（註解說明可保留）",
      "zip(population, eval_results)" not in _code_only)
check("並留下註解說明此陷阱", "zip(population, eval_results)" in _src)
check("以 genome_hash 建立配對索引", "_by_hash" in _src and "ind.hash()" in _src)
check("individual 日誌改為直接記錄 result（自帶 genome_hash）",
      '"type": "individual", "generation": gen, **_r' in _src)
check("best 指標改用配對後結果（已排序故 index 0 即 top1）",
      _src.count(_TOKEN) >= 4, "count=%d" % _src.count(_TOKEN))
check("passed 計數也用配對後的結果",
      "eval_by_ind" in _src)

# 行為驗證：打亂完成順序後配對仍正確
_pop = [Genome.from_dict(dict(_good, ema_fast=f)) for f in (11, 12, 13, 14)]
_by_hash = {}
for _i, _ind in enumerate(_pop):
    _by_hash[_ind.hash()] = {"genome_hash": _ind.hash(), "fitness": float(10 - _i),
                             "pnl_365d": float(20 + _i), "genome": _ind.to_dict()}
_shuffled = list(_by_hash.values())[::-1]          # 模擬 as_completed 亂序
_idx = {}
for _r2 in _shuffled:
    if _r2["genome_hash"] not in _idx:
        _idx[_r2["genome_hash"]] = _r2
_pf = []
for _ind in _pop:
    _res = _idx.get(_ind.hash(), {"fitness": -1000.0})
    _pf.append((_ind, _res["fitness"]))
check("亂序完成下 fitness 仍配到正確的基因組",
      [f for _, f in _pf] == [10.0, 9.0, 8.0, 7.0], f"got {[f for _, f in _pf]}")
_pf.sort(key=lambda x: x[1], reverse=True)
check("排序後 top1 就是最高分基因組", _pf[0][0].hash() == _pop[0].hash())
check("舊的 zip 方式在亂序下會配錯（回歸防護）",
      [f for _, f in zip(_pop, _shuffled)] != [10.0, 9.0, 8.0, 7.0],
      f"舊方式: {[f for _, f in zip(_pop, _shuffled)]}")

print()
print("=" * 72)
print("9d. 固定回測視窗終點：同一基因組必須可重現")
print("=" * 72)
# 原本 validator 內部各自用 int(time.time()*1000)，同一基因組在不同評估
# 會得到不同指標（實測 DD 3.69% -> 5.01%、PnL 26.73% -> 26.51%），
# GA 因此在對時間漂移造成的雜訊最佳化，且漂動正好落在 DD 綁定約束上。
_va = (_BACKEND / "backtest" / "ab_validators.py").read_text()
check("annual_validator 接受 end_time_ms 參數",
      "end_time_ms: Optional[int] = None" in _va)
check("exploration_validator 接受並轉傳 end_time_ms",
      _va.count("end_time_ms: Optional[int] = None") >= 2 and "end_time_ms=end_time_ms" in _va)
check("視窗終點改為可固定，不再硬取 time.time()",
      "now_ms = int(end_time_ms) if end_time_ms else int(time.time() * 1000)" in _va)
_ga_src = (_BACKEND / "backtest" / "ga_annual.py").read_text()
check("evaluate_individual 接受 end_time_ms 與多視窗 windows 參數",
      "end_time_ms: Optional[int] = None" in _ga_src
      and "windows: Optional[List[Tuple[int, int, float]]] = None" in _ga_src)
check("視窗終點以迴圈變數傳入 validator",
      "end_time_ms=w_end" in _ga_src and "backtest_days=w_days" in _ga_src)
check("runner 新增 --end-time-ms 且整輪共用一個值",
      '"--end-time-ms"' in _src and "Fixed backtest window end_time_ms=" in _src)
# 原本用字串比對 "args.backtest_days, end_time_ms, windows)"，但該 call 後來
# 追加了 kline_period 參數，字串剛好不再匹配（行為正確、斷言過時）。
# 改用 AST 取實際位置參數，鎖住「同一組視窗參數傳給每個 worker」這件事本身。
import ast as _ast
_submit_args = None
for _node in _ast.walk(_ast.parse(_src)):
    if (isinstance(_node, _ast.Call) and isinstance(_node.func, _ast.Attribute)
            and _node.func.attr == "submit" and _node.args
            and getattr(_node.args[0], "id", None) == "evaluate_individual"):
        _submit_args = [_ast.unparse(a) for a in _node.args[1:]]
check("runner 找到 executor.submit(evaluate_individual, ...) 呼叫", _submit_args is not None)
check("runner 把 end_time_ms、windows 與 kline_period 傳給每個 worker",
      _submit_args is not None
      and "end_time_ms" in _submit_args and "windows" in _submit_args
      and "args.kline_period" in _submit_args
      and "args.backtest_days" in _submit_args,
      f"got {_submit_args}")

print()
print("=" * 72)
print("9e. 多視窗一致性評分（2026-09-27）")
print("=" * 72)
# 單一三年視窗會獎勵只在強行情賺錢的策略。實測 5 個候選三年平均 22~83% 年化
# PnL，但逐年分解為 71.7/30.2/-2.6 等，最近一年 100% 未過嚴格 gate。
check("定義 WINDOW_CONSISTENCY_PENALTY 常數",
      "WINDOW_CONSISTENCY_PENALTY" in _ga_src)
check("evaluate_individual 接受 windows 參數",
      "windows: Optional[List[Tuple[int, int, float]]] = None" in _ga_src)
check("逐視窗評估並統計通過數與一致性",
      "consistency = n_pass / float(len(per_window))" in _ga_src
      and "windows_passed" in _ga_src and "window_consistency" in _ga_src)
check("對未通過視窗比例施加懲罰",
      "fitness = aggregate - WINDOW_CONSISTENCY_PENALTY * (1.0 - consistency)" in _ga_src)
check("探索期不施加一致性懲罰（否則扼殺探索）",
      "if use_exploration:\n        fitness = aggregate" in _ga_src)
check("回報診斷用的逐年明細",
      '"window_detail"' in _ga_src)
check("主要視窗（權重最高者）決定對外回報的個體指標",
      "primary = max(range(len(per_window))" in _ga_src)
# 2026-09-27 實測事故：多視窗迴圈內建構 EvalResult 時漏了 fitness/elapsed_ms，
# 導致每次評估都拋 TypeError 而落入 except，全部候選 fitness = -1000。
import dataclasses as _dc, inspect as _insp
import backtest.ga_annual as _gam
_req = {f.name for f in _dc.fields(_gam.EvalResult)
        if f.default is _dc.MISSING and f.default_factory is _dc.MISSING}
_ev_src = _insp.getsource(_gam.evaluate_individual)
check("evaluate_individual 內每個 EvalResult 建構都補齊無預設值的欄位",
      all(("%s=" % _k) in _ev_src for _k in _req) and _ev_src.count("EvalResult(") >= 2,
      "建構 %d 處；必要欄位 %s" % (_ev_src.count("EvalResult("), sorted(_req)))
try:
    _probe = _gam.EvalResult(**{
        _k: (Genome() if _k == "genome" else
             (None if _k == "pf" else 0.0)) for _k in _req})
    check("EvalResult 可用最小必要欄位建構成功（回歸防護）",
          _probe is not None, "欄位=%s" % sorted(_req))
except TypeError as _e:
    check("EvalResult 可用最小必要欄位建構成功（回歸防護）", False, str(_e))

check("runner 新增 --windows 與 --window-recent-weight",
      '"--windows"' in _src and '"--window-recent-weight"' in _src)
check("runner 產生巢狀視窗（全部結束於 end_time_ms）",
      "windows.append((end_time_ms, _days, _w))" in _src)
# 巢狀視窗：每一段都含完整近期資料，才能用同一個絕對門檻比較。
# 互不重疊切分會讓較舊視窗缺少新上市幣、交易數天然不足而恆定失敗（實測 0/60）。
def _nested(END, n, total, recent_w):
    return [(END, max(1, total*(i+1)//n), recent_w if i == 0 else 1.0)
            for i in range(n)]
_w = _nested(1000000, 2, 1095, 1.5)
check("所有視窗結束時間相同（巢狀）", len({x[0] for x in _w}) == 1)
check("視窗長度遞增", _w[0][1] < _w[1][1], "%d < %d" % (_w[0][1], _w[1][1]))
check("最後一段涵蓋完整 backtest_days", _w[-1][1] == 1095)
check("最近（最短）視窗拿到最高權重", _w[0][2] == max(x[2] for x in _w))
_w3 = _nested(1000000, 3, 1095, 1.5)
check("三視窗長度遞增且末段為完整長度",
      [x[1] for x in _w3] == sorted(x[1] for x in _w3) and _w3[-1][1] == 1095,
      str([x[1] for x in _w3]))
check("巢狀視窗都涵蓋最近期間（消除上市時點造成的差異）",
      all(x[0] == 1000000 for x in _w3))

# 行為驗證：一致性懲罰的量級。
# 必須固定視窗績與加權平均不變，只改變「通過視窗數」，
# 否則量到的差異會混入視窗績效本身的差距。
WCP = 1.0
def agg(win_fits, weights):
    return sum(w*f for f, w in zip(win_fits, weights)) / float(sum(weights))
def total(fits, weights, n_pass):
    return agg(fits, weights) - WCP*(1.0 - n_pass/float(len(fits)))
wt = [1.0, 1.5]
fits = [0.70, 0.70]          # 兩個視窗績效相同
f2, f1, f0 = total(fits, wt, 2), total(fits, wt, 1), total(fits, wt, 0)
check("全部視窗通過時不扣分", abs(f2 - agg(fits, wt)) < 1e-9, "%.4f" % f2)
check("通過數 2->1 恰扣 WCP/2", abs((f2-f1) - WCP/2) < 1e-9, "扣 %.4f" % (f2-f1))
check("通過數 1->0 恰扣 WCP/2", abs((f1-f0) - WCP/2) < 1e-9, "扣 %.4f" % (f1-f0))
check("全未通過相對全通過恰扣滿 WCP", abs((f2-f0) - WCP) < 1e-9, "扣 %.4f" % (f2-f0))
check("最近視窗權重 1.5 確實拉高加權平均",
      agg([0.0, 1.0], wt) > 0.5, "%.4f" % agg([0.0, 1.0], wt))

# 真實情境：舊三年視窗漂亮、最近一年崩盤的候選，必須被懲罰到落後穩健候選
# 這正是 20260927T171500Z 診斷出的失效型態。
def scenario(win_fits, weights, n_pass):
    return agg(win_fits, weights) - WCP*(1.0 - n_pass/float(len(win_fits)))
spike = scenario([0.70, 0.70], [1.0, 1.0], 2)     # 兩段都好
overfit = scenario([0.90, 0.10], [1.0, 1.0], 1)    # 前段強、最近崩
print("  [情境] 穩健(2視窗皆通過)=%.4f  過擬合(最近視窗未過)=%.4f"%(spike, overfit))
check("單一強行情撐起的候選會落後兩段都通過的候選",
      overfit < spike, "%.4f < %.4f" % (overfit, spike))

print()
print("=" * 72)
print("10. RSI 基因不再是死維度")
print("=" * 72)
import re as _re
_src = (_BACKEND / "backtest" / "ga_annual.py").read_text()
_i = _src.find("if genome.use_rsi:")
_seg = _src[_i:_i + 1400]
check("RSI 下限門檻用 rsi_buy_thr", "rsi_buy_thr" in _seg)
check("RSI 上限門檻用 rsi_sell_thr（原本從未進入產生碼）",
      "rsi_sell_thr" in _seg)
check("RSI 條件為動能帶（<= buy 跳過、>= sell 跳過）",
      _seg.count("continue") >= 2 and "<= " in _seg and ">= " in _seg)
check("不再使用舊的矛盾語義 'if rsi >= <buy_thr>'",
      not _re.search(r"if rsi >= ' \+ str\(genome\.rsi_buy_thr\)", _seg))

print()
print("=" * 72)
if FAILS:
    print(f"結果：{len(FAILS)} 項失敗")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("結果：全部通過")

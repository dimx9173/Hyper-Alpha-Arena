"""§7 收益驗證三關的純函數實作，供 Handoff Gate 使用。

移植自 QuantDinger 的 ``strategy_evolution/statistics.py``，逐行保留公式與
數值語義。移植而非 import，因為 Hyper 與 QD 是兩個獨立部署，執行期不該互相
依賴。一致性由 ``test_handoff_gates.py`` 的逐筆對帳測試守住（容差 1e-9）。

三個函數
--------
``deflated_sharpe``      第一關：降維夏普比率。回答「這個 Sharpe 有多少是
                         多重試驗選出來的」。
``probability_of_backtest_overfitting``
                         組合對稱交叉驗證的過擬合機率。回答「最優參數是不是
                         只在樣本內最優」。
``monte_carlo_bootstrap`` 第二關：區塊拔靴壓力測試。回答「換一條路徑，最大
                         回撤會多深」。

單位約定
--------
* ``returns`` 是逐期小數報酬（0.01 = 1%），不是百分比。
* ``observed_sharpe`` 必須與 ``returns`` 同一尺度（逐期，未年化）。QD 的呼叫
  端就是用 ``sample_sharpe(returns)`` 餵進來，本模組保持同一契約。
* 所有回傳值都是純 dict，可直接序列化進 ledger 或報告。

不可用時一律回 ``available: False`` 加 ``reason``，不假造 0 當成「已測量」。
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist, mean, median, pstdev
from typing import Any, Dict, Iterable, List, Sequence

# 與 QD 相同的常數；改動會讓對帳測試失敗。
EULER_GAMMA = 0.5772156649
_VARIATION_EPS = 1e-12
_DENOM_FLOOR = 1e-12
_ROUND = 6


def deflated_sharpe(
    returns: Sequence[float],
    *,
    observed_sharpe: float,
    trials: int,
) -> Dict[str, Any]:
    """降維夏普比率（Bailey & López de Prado）。

    ``trials`` 是候選策略總數。搜尋 100 個參數組合後拿到 Sharpe 1.0 與直接
    拿到 Sharpe 1.0 不是同一件事，這個函數把兩者的差距量化成機率。
    """
    values = [float(value) for value in returns if math.isfinite(float(value))]
    sample_count = len(values)
    if sample_count < 3:
        return {
            "available": False,
            "reason": "insufficientObservations",
            "probability": 0.0,
            "benchmarkSharpe": 0.0,
            "observations": sample_count,
        }
    avg = mean(values)
    sigma = pstdev(values)
    if sigma <= _VARIATION_EPS:
        return {
            "available": False,
            "reason": "insufficientVariation",
            "probability": 0.0,
            "benchmarkSharpe": 0.0,
            "observations": sample_count,
        }
    skew = mean(((value - avg) / sigma) ** 3 for value in values)
    kurtosis = mean(((value - avg) / sigma) ** 4 for value in values)
    effective_trials = max(1, int(trials))
    denominator = math.sqrt(
        max(
            _DENOM_FLOOR,
            1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2,
        )
    )
    if effective_trials == 1:
        benchmark = 0.0
    else:
        normal = NormalDist()
        sharpe_standard_error = denominator / math.sqrt(sample_count - 1)
        benchmark = sharpe_standard_error * (
            (1.0 - EULER_GAMMA) * normal.inv_cdf(1.0 - 1.0 / effective_trials)
            + EULER_GAMMA * normal.inv_cdf(1.0 - 1.0 / (effective_trials * math.e))
        )
    statistic = (observed_sharpe - benchmark) * math.sqrt(sample_count - 1) / denominator
    return {
        "available": True,
        "probability": round(NormalDist().cdf(statistic), _ROUND),
        "observedSharpe": round(observed_sharpe, _ROUND),
        "benchmarkSharpe": round(benchmark, _ROUND),
        "observations": sample_count,
        "skew": round(skew, _ROUND),
        "kurtosis": round(kurtosis, _ROUND),
    }


def probability_of_backtest_overfitting(
    score_matrix: Sequence[Sequence[float]],
    *,
    max_combinations: int = 2000,
) -> Dict[str, Any]:
    """PBO：把期間切成兩半，看樣本內最優者在樣本外的排名。

    每個 logit <= 0 代表這一輪切分中，樣本內最優者在樣本外落到中位數之後。
    PBO 是這種情形的比例。``score_matrix`` 是 [候選][期間] 的分數表。
    """
    matrix = [row for row in score_matrix if len(row) >= 4]
    if len(matrix) < 2:
        return {
            "available": False,
            "reason": "insufficientCandidates",
            "probability": 0.0,
            "samples": 0,
            "logits": [],
        }
    periods = min(len(row) for row in matrix)
    if periods < 4:
        return {
            "available": False,
            "reason": "insufficientFolds",
            "probability": 0.0,
            "samples": 0,
            "logits": [],
        }
    half = periods // 2
    combinations = list(_combinations(range(periods), half))
    if len(combinations) > max_combinations:
        rng = random.Random(1776)
        combinations = rng.sample(combinations, max_combinations)
    logits: List[float] = []
    for train_indexes in combinations:
        train_set = set(train_indexes)
        test_indexes = [index for index in range(periods) if index not in train_set]
        train_scores = [mean(row[index] for index in train_indexes) for row in matrix]
        winner = max(range(len(matrix)), key=lambda index: train_scores[index])
        test_scores = [mean(row[index] for index in test_indexes) for row in matrix]
        ordered = sorted(test_scores)
        rank = ordered.index(test_scores[winner]) + 1
        percentile = min(1.0 - 1e-9, max(1e-9, rank / (len(matrix) + 1.0)))
        logits.append(math.log(percentile / (1.0 - percentile)))
    probability = sum(1 for value in logits if value <= 0.0) / len(logits) if logits else 0.0
    return {
        "available": bool(logits),
        "probability": round(probability, _ROUND),
        "samples": len(logits),
        "medianLogit": round(median(logits), _ROUND) if logits else 0.0,
        "logits": [round(value, 5) for value in logits[:200]],
    }


def monte_carlo_bootstrap(
    returns: Sequence[float],
    *,
    paths: int,
    block_size: int,
    seed: int,
) -> Dict[str, Any]:
    """區塊拔靴：重抽報酬序列，量測末端報酬與最大回撤的分佈。

    區塊長度取 ``min(block_size, n^(1/3), n-1)``。``n^(1/3)`` 保留短期自相關，
    純 iid 重抽會低估回撤。
    """
    samples = [float(value) for value in returns if math.isfinite(float(value))]
    if len(samples) < 2:
        return {
            "available": False,
            "reason": "insufficientObservations",
            "paths": 0,
            "terminalReturns": [],
            "percentiles": {},
            "drawdownPercentiles": {},
            "lossProbability": 0.0,
        }
    if pstdev(samples) <= _VARIATION_EPS:
        return {
            "available": False,
            "reason": "insufficientVariation",
            "paths": 0,
            "terminalReturns": [],
            "percentiles": {},
            "drawdownPercentiles": {},
            "lossProbability": 0.0,
        }
    rng = random.Random(seed)
    terminal: List[float] = []
    drawdowns: List[float] = []
    length = len(samples)
    adaptive_block = max(1, int(round(length ** (1.0 / 3.0))))
    block = max(1, min(block_size, adaptive_block, length - 1))
    for _ in range(paths):
        path: List[float] = []
        while len(path) < length:
            start = rng.randrange(length)
            path.extend(samples[(start + offset) % length] for offset in range(block))
        equity = 1.0
        peak = 1.0
        worst = 0.0
        for value in path[:length]:
            equity *= 1.0 + value
            peak = max(peak, equity)
            worst = min(worst, (equity / peak - 1.0) * 100.0)
        terminal.append((equity - 1.0) * 100.0)
        drawdowns.append(worst)
    return {
        "available": True,
        "paths": paths,
        "observations": length,
        "requestedBlockSize": block_size,
        "effectiveBlockSize": block,
        "uniqueTerminalReturns": len({round(value, 10) for value in terminal}),
        "terminalReturns": _histogram(terminal, bins=24),
        "percentiles": _percentiles(terminal),
        "drawdownPercentiles": _percentiles(drawdowns),
        "lossProbability": round(sum(1 for value in terminal if value < 0) / paths, _ROUND),
        "ruinProbability30": round(sum(1 for value in terminal if value <= -30.0) / paths, _ROUND),
    }


def sample_sharpe(returns: Iterable[float]) -> float:
    """逐期 Sharpe（未年化），與 ``deflated_sharpe`` 的 ``observed_sharpe`` 同尺度。"""
    values = [float(value) for value in returns if math.isfinite(float(value))]
    if len(values) < 2:
        return 0.0
    sigma = pstdev(values)
    return mean(values) / sigma if sigma > _VARIATION_EPS else 0.0


def _histogram(values: List[float], *, bins: int) -> List[Dict[str, float]]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [{"value": round(low, 4), "count": len(values)}]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    return [
        {"value": round(low + (index + 0.5) * width, 4), "count": count}
        for index, count in enumerate(counts)
    ]


def _percentiles(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        str(item): round(_quantile(ordered, item / 100.0), 4)
        for item in (1, 5, 25, 50, 75, 95, 99)
    }


def _quantile(values: List[float], probability: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _combinations(values: Iterable[int], size: int):
    values = tuple(values)
    if size == 0:
        yield ()
        return
    for index in range(len(values)):
        for tail in _combinations(values[index + 1:], size - 1):
            yield (values[index],) + tail


def handoff_gate(
    trade_returns: Sequence[float],
    *,
    trials: int = 100,
    mc_paths: int = 1000,
    mc_block_size: int = 8,
    mc_seed: int = 42,
    dsr_min: float = 0.95,
    max_drawdown_pct: float = 20.0,
    ruin_max_prob: float = 0.005,
) -> Dict[str, Any]:
    """三關一次判：DSR、MC 回撤、（PBO 由呼叫端另行提供分數表）。

    ``trade_returns`` 是 FuxiBacktestResult.trade_returns() 的逐筆淨報酬。
    MC 報的是百分比口徑（與回測報表一致），DSR 吃逐期小數口徑。
    """
    obs = sample_sharpe(trade_returns)
    dsr = deflated_sharpe(list(trade_returns), observed_sharpe=obs, trials=trials)
    mc = monte_carlo_bootstrap(
        list(trade_returns), paths=mc_paths, block_size=mc_block_size, seed=mc_seed
    )
    dsr_pass = bool(dsr.get("available")) and float(dsr.get("probability", 0.0)) >= dsr_min
    # drawdowns 全為負值，升序的 p95 是最樂觀的一端。尾部風險看 p5：
    # 95% 的路徑都比它好，這才是規格的「第 95 百分位極端最大回撤」。
    dd_tail = float((mc.get("drawdownPercentiles") or {}).get("5", 0.0) or 0.0)
    dd_pass = bool(mc.get("available")) and abs(dd_tail) <= max_drawdown_pct
    ruin_prob = float(mc.get("ruinProbability30", 1.0) or 0.0)
    ruin_pass = bool(mc.get("available")) and ruin_prob <= ruin_max_prob
    passed = bool(dsr_pass and dd_pass and ruin_pass)
    return {
        "passed": passed,
        "dsr": dsr,
        "dsr_pass": dsr_pass,
        "mc": {"available": mc.get("available"), "paths": mc.get("paths"),
               "drawdownPercentiles": mc.get("drawdownPercentiles"),
               "dd_tail_p5": round(dd_tail, 4),
               "lossProbability": mc.get("lossProbability"),
               "ruin_probability": round(ruin_prob, 6)},
        "dd_pass": dd_pass,
        "ruin_pass": ruin_pass,
    }

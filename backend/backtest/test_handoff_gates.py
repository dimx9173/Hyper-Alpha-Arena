"""handoff_gates 回歸：與 QD statistics.py 逐筆對帳，容差 1e-9。

FAIL 條件：任何一個函數的輸出與 QD 同輸入的結果差超過 1e-9，
代表移植時改了公式，移植即無效。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

QD_STATS = Path(
    "/Users/carlos/Projects/QuantDinger/backend_api_python/app/services/strategy_evolution/statistics.py"
)

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from backtest import handoff_gates as hg


def _load_qd():
    spec = importlib.util.spec_from_file_location("qd_statistics", QD_STATS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_QD = _load_qd()


def _returns(n: int = 200, seed: int = 7):
    import random

    rng = random.Random(seed)
    return [rng.gauss(0.001, 0.02) for _ in range(n)]


def _assert_close_mine(a, b, tol=1e-9, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} vs {set(b)}"
        for k in a:
            _assert_close_mine(a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close_mine(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert abs(a - b) <= tol, f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def test_dsr_matches_qd():
    rets = _returns()
    from statistics import mean, pstdev

    obs = mean(rets) / pstdev(rets)
    mine = hg.deflated_sharpe(rets, observed_sharpe=obs, trials=100)
    theirs = _QD.deflated_sharpe(rets, observed_sharpe=obs, trials=100)
    assert set(mine) == set(theirs)
    _assert_close_mine(mine, theirs)


def test_dsr_unavailable_matches_qd():
    for rets in ([], [0.01], [0.0, 0.0, 0.0, 0.0]):
        mine = hg.deflated_sharpe(list(rets), observed_sharpe=0.0, trials=10)
        theirs = _QD.deflated_sharpe(list(rets), observed_sharpe=0.0, trials=10)
        assert mine == theirs, f"{rets}: {mine} vs {theirs}"


def test_pbo_matches_qd():
    import random

    rng = random.Random(11)
    matrix = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(6)]
    mine = hg.probability_of_backtest_overfitting(matrix, max_combinations=500)
    theirs = _QD.probability_of_backtest_overfitting(matrix, max_combinations=500)
    assert set(mine) == set(theirs)
    assert mine["samples"] == theirs["samples"]
    _assert_close_mine(mine["probability"], theirs["probability"])
    _assert_close_mine(mine["medianLogit"], theirs["medianLogit"])
    assert mine["logits"] == theirs["logits"]


def test_mc_matches_qd():
    rets = _returns(120, seed=13)
    mine = hg.monte_carlo_bootstrap(rets, paths=50, block_size=8, seed=42)
    theirs = _QD.monte_carlo_bootstrap(rets, paths=50, block_size=8, seed=42)
    # mine 多一個 ruinProbability30（handoff 門專用），對帳只比共有鍵。
    assert set(theirs) <= set(mine)
    for key in theirs:
        _assert_close_mine(mine[key], theirs[key], path=f"mc.{key}")
    assert mine["terminalReturns"] == theirs["terminalReturns"]
    assert mine["paths"] == theirs["paths"]
    assert mine["effectiveBlockSize"] == theirs["effectiveBlockSize"]


def test_mc_unavailable_matches_qd():
    mine = hg.monte_carlo_bootstrap([0.01], paths=10, block_size=4, seed=1)
    theirs = _QD.monte_carlo_bootstrap([0.01], paths=10, block_size=4, seed=1)
    assert mine == theirs

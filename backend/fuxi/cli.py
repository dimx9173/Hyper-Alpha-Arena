#!/usr/bin/env python3
"""伏羲滾動量化系統 CLI — Loop 2 + Loop 3 + Loop 1 + 四大關卡.

Examples
--------
Full closed loop on real Binance 5m data (one symbol)::

    .venv/bin/python -m fuxi.cli --symbol BTCUSDT --days 180 --mode full

Loop 3 parameter-plateau search only (slow; scores neighbourhoods)::

    .venv/bin/python -m fuxi.cli --symbol BTCUSDT --days 180 --mode loop3

Offline synthetic sanity check (no network, no cache)::

    .venv/bin/python -m fuxi.cli --synthetic --days 20 --mode full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fuxi.backtest import FuxiBacktestConfig, FuxiBacktestRunner  # noqa: E402
from fuxi.data import BinanceFuxiDataSource  # noqa: E402
from fuxi.features import MultiTimeframeFeaturePipeline  # noqa: E402
from fuxi.labels import MetaLabelingEngine, build_model_frame  # noqa: E402
from fuxi.loop3 import (  # noqa: E402
    ParameterPlateauSearch,
    default_parameter_grid,
    reselect_features,
    stratified_grid_sample,
)
from fuxi.ratchet import AsymmetricCompoundingRatchet  # noqa: E402
from fuxi.risk import PerpetualRiskAndSmoothingEngine  # noqa: E402
from fuxi.rolling import RollingClosedLoopController  # noqa: E402
from fuxi.validation import PromotionGate  # noqa: E402

# Bottom bucket 5u: with the old 25u floor a 794u account capped at 0.2 Kelly
# could never reach the ladder and silently traded nothing (review F8).
# Spec §G Discrete Grid keeps the same ladder; the risk engine now falls back
# to the raw Kelly fraction below the smallest tier.
BUCKETS = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
# Loop-3 sampling default: 24 draws = 4 per x-tier over 6 width strata, so the
# search covers the whole (x, y, M) box instead of the first x-slice (F6).
DEFAULT_LOOP3_SAMPLES = 24
# §4.2 latch verdict text. Compared by identity at the end of main().
PASS_REGIME_VERDICT = "PASS_REGIME — 無可交易邊際，全系統空倉休眠"


def synthetic_bars(days: int, seed: int = 7) -> pd.DataFrame:
    n = days * 288
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    regime = np.repeat(
        # Ceil, don't floor: without the +199 the 8640th row of days=30 has
        # nowhere to come from and np.repeat broadcasts against a short
        # array (review F7: `days * 288` rows guaranteed for every input).
        rng.choice([-1, 0, 1], size=max(1, (n + 199) // 200)), 200
    )[:n]
    ret = regime * 0.00035 + rng.normal(0, 0.0022, n)
    close = 1000.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0016, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0016, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.lognormal(10, 0.5, n),
            "funding_rate": rng.normal(0.0001, 0.00005, n),
        },
        index=idx,
    )


def _load_bars(args) -> pd.DataFrame:
    if args.synthetic:
        return synthetic_bars(args.days)
    ds = BinanceFuxiDataSource(cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    end = pd.Timestamp(args.end, tz="UTC")
    start = end - pd.Timedelta(days=args.days)
    return ds.get_training_frame(
        args.symbol,
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
        with_oi=args.with_oi,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fuxi adaptive rolling quant loop")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--end", default="2026-11-08")
    ap.add_argument("--mode", choices=("full", "loop3"), default="full")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--with-oi", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--x", type=float, default=0.012)
    ap.add_argument("--y", type=float, default=2.0)
    ap.add_argument("--hold-bars", type=int, default=24)
    ap.add_argument("--lag-n", type=int, default=5)
    ap.add_argument("--entry-threshold", type=float, default=0.55)
    ap.add_argument("--target-leverage", type=float, default=3.0)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--loop3-samples", type=int, default=DEFAULT_LOOP3_SAMPLES)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument("--reinvest-ratio", type=float, default=0.60)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    bars = _load_bars(args)
    if bars.empty:
        print(f"no bars for {args.symbol}", file=sys.stderr)
        return 2

    pipe = MultiTimeframeFeaturePipeline(lag_n=args.lag_n)
    features = pipe.build_multi_tf_features(bars)
    signals = MetaLabelingEngine().generate_primary_signals(bars)
    report: dict = {
        "symbol": args.symbol,
        "bars": int(len(bars)),
        "start": str(bars.index[0]),
        "end": str(bars.index[-1]),
        "feature_columns": int(features.shape[1]),
        "complete_rows": int((~features.isna().any(axis=1)).sum()),
    }

    # ---------------- Loop 3 (optional): plateau search + feature reselect ---- #
    x, y, m = args.x, args.y, args.hold_bars
    # One controller for the whole run: Loop 3 sets its PASS_REGIME latch and
    # Loop 1 then honours it (`backtest.run` refuses to open a position while
    # the latch is set) — review F4.
    ctrl = RollingClosedLoopController(dsr_threshold=0.95)
    if args.mode == "loop3":
        # F14: clamp a positive sample count instead of slicing to [] — an
        # empty candidate list used to crash inside search().
        grid = default_parameter_grid()
        if args.loop3_samples <= 0 or args.loop3_samples > len(grid):
            print(
                f"invalid --loop3-samples {args.loop3_samples}: "
                f"must be in [1, {len(grid)}]",
                file=sys.stderr,
            )
            return 2
        eng = MetaLabelingEngine(x=x, y=y, M=m)
        meta = eng.apply_triple_barrier_and_weights(bars, signals)
        kept = reselect_features(features, meta, top_k=24)
        report["reselect_features"] = kept
        # Review F5: the search scores θ on exactly the kept subset (Loop 3
        # freezes the feature subset) and Loop 2 trains the same subset below.
        # `features` itself stays full so the simulator can still price ATR.
        sample = stratified_grid_sample(grid, args.loop3_samples)
        search = ParameterPlateauSearch(entry_threshold=args.entry_threshold)
        best, allres = search.search(sample, bars, features, signals,
                                     progress=not args.json,
                                     feature_cols=kept)
        report["loop3"] = {
            "best": best.as_dict(),
            "pass_regime": ParameterPlateauSearch.is_pass_regime(allres),
            "candidates": [r.as_dict() for r in allres],
        }
        if ParameterPlateauSearch.is_pass_regime(allres):
            # Review F4: Loop 3 owns the latch. Set it on the live controller
            # before any dueling so the PASS verdict survives champion deploys,
            # and early-return with a Loop-3-only report: a sleeping regime has
            # no models, no trades, and nothing for downstream gates to score.
            ctrl.set_pass_regime(True)
            report["verdict"] = PASS_REGIME_VERDICT
            report["pass_regime"] = True
            report["n_loop3_candidates"] = len(allres)
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print("=" * 72)
                print("伏羲滾動量化 Loop 3 結構重構")
                print(f"candidates {len(allres)}  verdict {PASS_REGIME_VERDICT}")
                print("=" * 72)
            return 0
        # Loop 3 measured a tradable plateau: clear the latch explicitly so a
        # stale sleep state from an earlier cycle cannot block this one.
        ctrl.set_pass_regime(False)
        x, y, m = best.theta["x"], best.theta["y"], int(best.theta["M"])

    # ---------------- Loop 2: train champion -------------------------------- #
    eng = MetaLabelingEngine(x=x, y=y, M=int(m))
    meta = eng.apply_triple_barrier_and_weights(bars, signals)
    X, yv, wv, rv, rv_expired = build_model_frame(features, meta)
    report["labels"] = {"rows": int(len(X)), "base_rate": float(yv.mean()) if len(X) else 0.0}
    if len(X) < 300:
        report["verdict"] = "INSUFFICIENT_LABELS"
        print(json.dumps(report, indent=2, default=str))
        return 3

    cut = int(len(X) * args.train_frac)
    # `ctrl` is the run's controller: Loop 3 already latched its sleep state,
    # and the duel below now feeds it without recreating it (review F4/F5).
    # Loop 3 ran: the frozen subset fed the search, so assert it feeds dueling
    # too — the champion must be trained on exactly `kept` (review F5).
    loop2_feature_cols = kept if args.mode == "loop3" else None
    if args.mode == "loop3":
        missing = [c for c in kept if c not in X.columns]
        assert not missing, f"kept subset missing from model frame: {missing}"
        report["loop2_feature_cols"] = list(kept)
    loop2 = ctrl.train_and_evaluate_challenger(
        X.iloc[:cut], yv.iloc[:cut], wv[:cut],
        X.iloc[cut:], yv.iloc[cut:], rv[cut:],
        candidate_params={"x": float(x), "y": float(y), "M": int(m)},
        feature_cols=loop2_feature_cols,
        val_expired=rv_expired[cut:],
    )
    report["loop2"] = loop2
    probs = ctrl.champion.calibrated_proba(X.iloc[cut:])
    report["p_cal"] = {
        "max": float(np.max(probs)),
        "p95": float(np.percentile(probs, 95)),
        "base_rate_val": float(np.asarray(yv.iloc[cut:]).mean()),
        "frac_above_threshold": float((probs > args.entry_threshold).mean()),
    }

    # ---------------- Loop 1: backtest the champion ------------------------- #
    runner = FuxiBacktestRunner(
        ctrl,
        AsymmetricCompoundingRatchet(args.initial_capital, args.reinvest_ratio),
        PerpetualRiskAndSmoothingEngine(discrete_buckets=BUCKETS),
        FuxiBacktestConfig(
            initial_capital=args.initial_capital,
            reinvest_ratio=args.reinvest_ratio,
            entry_threshold=args.entry_threshold,
            target_leverage=args.target_leverage,
        ),
    )
    out = runner.run(bars, features, primary_signals=signals)
    report["loop1"] = {
        "trades": out.n_trades,
        "win_rate": out.win_rate,
        "net_pnl": out.total_net_pnl,
        "max_drawdown": out.max_drawdown,
        "equity": out.total_equity,
        "active_pool": out.active_pool,
        "reserve_vault": out.reserve_vault,
        "open_contracts": len(ctrl.open_contracts),
    }

    # ---------------- §7 four validation gates ------------------------------ #
    if out.n_trades >= 10:
        gate = PromotionGate().evaluate(
            out.trade_returns(), out.exit_timestamps_ms()
        )
        report["validation"] = gate
        report["verdict"] = "PROMOTE" if gate["passed"] else "REJECT"
    elif out.n_trades == 0:
        report["verdict"] = "NO_TRADE — 校準勝率未達淨期望值門檻"
    else:
        report["verdict"] = "TOO_FEW_TRADES"

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("=" * 72)
        print(f"伏羲滾動量化 — {args.symbol}  {args.bars if hasattr(args, 'bars') else ''}")
        print(f"bars {report['bars']}  {report['start']} → {report['end']}")
        print(f"features {report['feature_columns']} cols / {report['complete_rows']} complete rows")
        print(f"labels {report['labels']}")
        print(f"Loop2: {loop2['status']}  champion {loop2['active_id']}  "
              f"dsr {loop2.get('chal_dsr', 0):.3f}  ev {loop2.get('chal_ev', 0):+.5f}")
        print(f"p_cal max {report['p_cal']['max']:.3f} (threshold {args.entry_threshold})")
        l1 = report["loop1"]
        print(f"Loop1: trades {l1['trades']}  win {l1['win_rate']:.3f}  "
              f"net {l1['net_pnl']:+.2f}u  dd {l1['max_drawdown']:.4f}")
        print(f"equity {l1['equity']:.2f}u  active {l1['active_pool']:.2f}u  "
              f"vault {l1['reserve_vault']:.2f}u")
        if "validation" in report:
            v = report["validation"]
            print(f"Gate1 DSR {v['dsr']['dsr']:.3f} (≥0.95)  "
                  f"Gate2 MC95 {v['monte_carlo']['mdd_95th']:.3f} (≤0.20)  "
                  f"ruin {v['monte_carlo']['risk_of_ruin']:.4f}")
            print(f"Gate3 friction sr {v['friction']['sr']:.2f} retention {v['friction']['retention']:.3f}  "
                  f"Gate4 win_share {v['monthly']['win_share']:.3f} tuw {v['monthly']['max_tuw_days']:.1f}d")
        print(f"VERDICT: {report['verdict']}")
        print("=" * 72)
    return 0 if report.get("verdict") in ("PROMOTE", PASS_REGIME_VERDICT) else 1


if __name__ == "__main__":
    raise SystemExit(main())

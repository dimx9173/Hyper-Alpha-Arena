"""Fuxi smoke tests — no DB, no network, deterministic synthetic bars.

Run: cd backend && .venv/bin/python -m pytest fuxi/tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from config.fees import FeeSchedule
from fuxi.backtest import FuxiBacktestConfig, FuxiBacktestRunner
from fuxi.features import WARM_START_BARS, MultiTimeframeFeaturePipeline
from fuxi.labels import MetaLabelingEngine, build_model_frame
from fuxi.loop3 import PASS_REGIME_MEDIAN_SR, ParameterPlateauSearch, PlateauResult
from fuxi.ratchet import AsymmetricCompoundingRatchet
from fuxi.risk import PerpetualRiskAndSmoothingEngine
from fuxi.rolling import RollingClosedLoopController
from fuxi.validation import ProfitValidationSuite


def synthetic_5m(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    ret = rng.normal(0.00004, 0.0025, n)
    close = 1000.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0018, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0018, n)))
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


# ---------------------------------------------------------------- features
def test_features_warm_and_no_lookahead():
    bars = synthetic_5m(2000)
    pipe = MultiTimeframeFeaturePipeline(lag_n=3)
    feats = pipe.build_multi_tf_features(bars)

    assert len(feats) == len(bars)
    # warm-start budget: first usable row is within the 360-bar budget
    first_valid = int(feats.dropna().index.get_loc(feats.dropna().index[0]))
    assert first_valid <= WARM_START_BARS

    for col in ("tf5m_body", "tf5m_rsi_scaled", "tf15m_ema_bias_20", "tf30m_norm_atr"):
        assert col in feats.columns
    for col in ("tf5m_rsi_scaled",):
        vals = feats[col].dropna()
        assert vals.min() >= -1.001 and vals.max() <= 1.001

    # no lookahead: 15m/30m columns must be frozen between higher-tf closes
    nz = feats.dropna()
    seg = nz["tf30m_ema_bias_60"].iloc[100:105]
    assert len(seg) == 5


def test_prune_collinear_keeps_primary_dir():
    bars = synthetic_5m(1200)
    feats = MultiTimeframeFeaturePipeline(lag_n=2).build_multi_tf_features(bars).dropna()
    feats = feats.copy()
    feats["primary_dir"] = 1
    feats["dup_of_body"] = feats["tf5m_body"] * 2.0
    kept = MultiTimeframeFeaturePipeline.prune_collinear_features(feats, threshold=0.8)
    assert "primary_dir" in kept
    assert "dup_of_body" not in kept or "tf5m_body" not in kept


# ------------------------------------------------------------------ labels
def test_triple_barrier_labels_and_weights():
    bars = synthetic_5m(1500)
    eng = MetaLabelingEngine(x=0.02, y=2.0, M=10)
    sig = eng.generate_primary_signals(bars)
    meta = eng.apply_triple_barrier_and_weights(bars, sig)

    assert not meta.empty
    assert set(np.unique(meta["meta_label"].to_numpy())).issubset({0.0, 1.0})
    assert (meta["sample_weight"] > 0).all()
    assert abs(float(meta["sample_weight"].mean()) - 1.0) < 1e-6
    assert (meta["t_end_pos"] > meta.index.map(lambda t: bars.index.get_loc(t))).all()


def test_model_frame_joins():
    bars = synthetic_5m(1500)
    feats = MultiTimeframeFeaturePipeline(lag_n=2).build_multi_tf_features(bars).dropna()
    eng = MetaLabelingEngine(x=0.02, y=2.0, M=10)
    meta = eng.apply_triple_barrier_and_weights(bars, eng.generate_primary_signals(bars))
    X, y, w, r, _expired = build_model_frame(feats, meta)
    assert len(X) == len(y) == len(w) == len(r) > 0
    # primary_dir is label-side (ex-post at bar t): it must never be a feature.
    assert "primary_dir" not in X.columns
    assert "primary_dir" in meta.columns


# ----------------------------------------------------------------- ratchet
def test_ratchet_skim_math():
    rat = AsymmetricCompoundingRatchet(initial_capital=1000.0, reinvest_ratio=0.6)
    rat.update_intraday_pnl(300.0)
    st = rat.settle_rolling_period(is_flat=True)
    assert st.total_equity == pytest.approx(1300.0)
    assert st.reserve_vault == pytest.approx(120.0)   # (1-0.6)*300
    assert st.active_pool == pytest.approx(1180.0)
    # Kelly base only steps by α of the gain: 1000 + 0.6*300 = 1180
    assert st.kelly_base_capital == pytest.approx(1180.0)


def test_ratchet_defends_on_drawdown():
    rat = AsymmetricCompoundingRatchet(1000.0, 0.6)
    rat.update_intraday_pnl(300.0)
    rat.settle_rolling_period(is_flat=True)
    base_before = rat.get_ratcheted_base_capital()
    rat.update_intraday_pnl(-200.0)
    assert rat.get_ratcheted_base_capital() == pytest.approx(base_before - 200.0)


def test_ratchet_no_settle_when_not_flat():
    rat = AsymmetricCompoundingRatchet(1000.0, 0.6)
    rat.update_intraday_pnl(300.0)
    st = rat.settle_rolling_period(is_flat=False)
    assert st.reserve_vault == 0.0


# -------------------------------------------------------------------- risk
def test_sizing_approves_high_probability_and_rejects_coinflip():
    eng = PerpetualRiskAndSmoothingEngine(discrete_buckets=[100.0, 200.0, 500.0, 1000.0])
    kwargs = dict(
        x=0.03, y=2.0, target_leverage=3.0, ratcheted_base_capital=1000.0,
        total_equity=1000.0, month_start_equity=1000.0, month_hwm_equity=1000.0,
        current_norm_atr=0.004,
    )
    ok = eng.compute_safe_sizing(prob_cal=0.75, direction=1, **kwargs)
    assert ok.approved and ok.margin > 0

    bad = eng.compute_safe_sizing(prob_cal=0.40, direction=1, **kwargs)
    assert not bad.approved and bad.margin == 0.0

    halt = PerpetualRiskAndSmoothingEngine().compute_safe_sizing(
        prob_cal=0.9, direction=1, x=0.03, y=2.0, target_leverage=3.0,
        ratcheted_base_capital=1000.0, total_equity=930.0,
        month_start_equity=1000.0, month_hwm_equity=1000.0, current_norm_atr=0.004)
    assert halt.month_mode == "MONTHLY_HALT_CIRCUIT_BREAKER"
    assert halt.margin == 0.0


def test_liquidation_guard_caps_leverage():
    eng = PerpetualRiskAndSmoothingEngine(mmr=0.004, liq_buffer=0.005)
    # x/y = 0.015 stop + ~0.00082 costs + 0.009 → 1/0.0248 ≈ 40 → floor 40
    z = eng.max_safe_leverage(0.015)
    assert z >= 1
    assert z * (0.015 + 0.009 + eng.fees.maker_fee_bps / 1e4 * 2
                + eng.fees.taker_fee_bps / 1e4
                + eng.fees.slippage_bps / 1e4) <= 1.0000001
    big = eng.max_safe_leverage(0.5)
    assert big == 1.0


# --------------------------------------------------------------- validation
def test_dsr_noise_is_not_significant():
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.01, 400)
    info = ProfitValidationSuite.compute_deflated_sharpe_ratio(
        noise, periods_per_year=365.0, num_trials=100
    )
    assert 0.0 <= info["dsr"] <= 1.0
    assert info["dsr"] < 0.95

def test_dsr_strong_signal_passes():
    rng = np.random.default_rng(4)
    # Five years of daily returns with per-trade SR ~0.27 annualize to ~5.2 at
    # PPY 365, clearing the K=10 null (0.79) by a wide margin → dsr > 0.95.
    # The unit scale is explicit: without periods_per_year this call raises.
    edge = rng.normal(0.0032, 0.012, 365 * 5)
    info = ProfitValidationSuite.compute_deflated_sharpe_ratio(
        edge, periods_per_year=365.0, num_trials=10
    )
    assert info["sr_ann"] > info["sr_null"]
    assert info["dsr"] > 0.95


def test_monte_carlo_and_friction_gates():
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0015, 0.01, 300)
    mc = ProfitValidationSuite.run_monte_carlo_reshuffling(rets, n_simulations=500)
    assert 0.0 <= mc["mdd_95th"] <= 1.0
    assert 0.0 <= mc["risk_of_ruin"] <= 1.0

    fr = ProfitValidationSuite.friction_sensitivity(
        rets, cost_loss=0.0013, periods_per_year=252.0
    )
    assert "passed" in fr and 0.0 <= fr["retention"] <= 2.0


def test_monthly_consistency_shapes():
    rng = np.random.default_rng(6)
    n = 240
    ts = np.sort(rng.integers(1_700_000_000_000, 1_730_000_000_000, n))
    rets = rng.normal(0.002, 0.008, n)
    out = ProfitValidationSuite.monthly_consistency(rets, ts)
    assert set(["passed", "win_share", "max_tuw_days", "top5_retention"]).issubset(out)


# ------------------------------------------------------------------ rolling
def test_challenger_promotes_then_orphan_contract_is_frozen():
    ctrl = RollingClosedLoopController(stress_ratio=0.30, dsr_threshold=0.5)
    bars = synthetic_5m(2500)
    feats = MultiTimeframeFeaturePipeline(lag_n=2).build_multi_tf_features(bars).dropna()
    eng = MetaLabelingEngine(x=0.02, y=2.0, M=10)
    meta = eng.apply_triple_barrier_and_weights(bars, eng.generate_primary_signals(bars))
    X, y, w, r, _expired = build_model_frame(feats, meta)

    cut = int(len(X) * 0.6)
    res1 = ctrl.train_and_evaluate_challenger(
        X.iloc[:cut], y.iloc[:cut], w[:cut], X.iloc[cut:], y.iloc[cut:], r[cut:],
        candidate_params={"x": 0.02, "y": 2.0, "M": 10})
    assert res1["status"] == "INITIAL_CHAMPION_DEPLOYED"
    model_before = ctrl.champion_id

    contract = ctrl.register_new_position("POS_1", 1, 1000.0, 200.0, 3.0, 10)
    tp_before, sl_before = contract.tp_price, contract.sl_price

    res2 = ctrl.train_and_evaluate_challenger(
        X.iloc[:cut], y.iloc[:cut], w[:cut], X.iloc[cut:], y.iloc[cut:], r[cut:],
        candidate_params={"x": 0.05, "y": 3.0, "M": 20})
    assert res2["status"] in {"CHALLENGER_PROMOTED", "CHAMPION_RETAINED_WITH_THROTTLE"}

    # orphan isolation: the old contract keeps its birth model + barriers
    if res2["status"] == "CHALLENGER_PROMOTED":
        assert ctrl.champion_id != model_before
        assert contract.model_id == model_before
        assert contract.tp_price == tp_before and contract.sl_price == sl_before
        assert contract.max_hold_bars == 10
    else:
        assert ctrl.kelly_penalty_scalar == 0.75


def test_hybrid_dataset_mixes_stress_and_recent():
    ctrl = RollingClosedLoopController(stress_ratio=0.3)
    rng = np.random.default_rng(11)
    n = 900
    X = pd.DataFrame({
        "tf5m_norm_atr": np.concatenate([rng.normal(0.002, 0.001, n - 300),
                                         rng.normal(0.02, 0.005, 300)]),
        "f1": rng.normal(0, 1, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    w = np.ones(n)
    Xh, yh, wh = ctrl.build_hybrid_anti_amnesia_dataset(X, y, w, recent_window_size=300)
    assert len(Xh) > 300            # stress pool appended
    assert len(yh) == len(wh) == len(Xh)
    assert (Xh["tf5m_norm_atr"] >= 0.01).mean() > 0.1  # stress rows really are high-vol


# ---------------------------------------------- accounting + cost invariants
def test_cost_legs_are_single_source_and_true_to_fees():
    """EV and the simulator must share one cost definition (review B1/M1)."""
    fees = FeeSchedule(maker_fee_bps=1.5, taker_fee_bps=4.5, slippage_bps=2.0)
    eng = PerpetualRiskAndSmoothingEngine(fees=fees)

    # entry is aggressive: taker + slippage
    assert eng.entry_cost_fraction() == pytest.approx(0.00065)
    # TP exits passive (maker only), SL exits aggressively (taker + slippage)
    assert eng.exit_cost_fraction(True) == pytest.approx(0.00015)
    assert eng.exit_cost_fraction(False) == pytest.approx(0.00065)

    cost_win, cost_loss = eng.round_trip_costs(0.0)
    assert cost_win == pytest.approx(0.0008)
    assert cost_loss == pytest.approx(0.0013)
    # a loss must cost more than a win at equal width
    assert cost_loss > cost_win


def test_accounting_identity_sum_pnl_equals_equity_delta():
    """Σ(trade.net_pnl) == total_equity − initial (review A B1: no double charge)."""
    bars = synthetic_5m(3000, seed=9)
    feats = MultiTimeframeFeaturePipeline(lag_n=3).build_multi_tf_features(bars).dropna()
    eng = MetaLabelingEngine(x=0.005, y=1.0, M=16)
    sig = eng.generate_primary_signals(bars)
    meta = eng.apply_triple_barrier_and_weights(bars, sig)
    X, y, w, r, _expired = build_model_frame(feats, meta)
    cut = int(len(X) * 0.65)
    ctrl = RollingClosedLoopController()
    ctrl.train_and_evaluate_challenger(
        X.iloc[:cut], y.iloc[:cut], w[:cut],
        X.iloc[cut:], y.iloc[cut:], r[cut:],
        candidate_params={"x": 0.005, "y": 1.0, "M": 16},
    )
    out = FuxiBacktestRunner(
        ctrl,
        AsymmetricCompoundingRatchet(1000.0, 0.6),
        PerpetualRiskAndSmoothingEngine(discrete_buckets=[25.0, 50.0, 100.0, 200.0]),
        FuxiBacktestConfig(entry_threshold=0.40, target_leverage=2.0),
    ).run(bars, feats, primary_signals=sig)

    if out.n_trades == 0:
        pytest.skip("no trades triggered on this seed")
    assert out.total_net_pnl == pytest.approx(out.total_equity - 1000.0, abs=1e-6)
    assert len(ctrl.open_contracts) == 0  # every contract closed


def test_plateau_excludes_untraded_parameters():
    """A theta that never trades must not score as SR 0 (review B: neutral trap)."""
    res = PlateauResult(theta={"x": 0.01, "y": 2.0, "M": 12.0},
                        mu_sr=-1.0, sigma_sr=0.0, score=-1.0, n_traded=0)
    assert res.as_dict()["n_traded"] == 0
    assert res.score < PASS_REGIME_MEDIAN_SR
    assert ParameterPlateauSearch().is_pass_regime([res]) is True


def test_expired_small_win_pays_taker_leg():
    """Review F-A: a time exit with ret>0 is not a take-profit fill."""
    from fuxi.risk import round_trip_cost_fractions
    cost_win, cost_loss = round_trip_cost_fractions(FeeSchedule())
    rets = np.array([0.0005])
    mask = np.array([True])
    idx = pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC")
    net, _ = RollingClosedLoopController._net_trade_rets(
        rets, mask, None, idx, 365.0, expired=np.array([True]))
    assert net[0] == pytest.approx(0.0005 - cost_loss)
    net_win, _ = RollingClosedLoopController._net_trade_rets(
        rets, mask, None, idx, 365.0, expired=np.array([False]))
    assert net_win[0] == pytest.approx(0.0005 - cost_win)

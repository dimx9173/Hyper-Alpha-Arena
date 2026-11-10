"""§6.1 實盤／紙上單逐根驅動器：把狀態機接上每根 5m 收盤。

在此之前 ``PositionStateMachine`` 與 ``PaperBrokerAdapter`` 只有測試呼叫端，
沒有任何生產路徑按根驅動它們。本模組補上這一塊：``run_paper_symbol`` 每根
收盤依序做

1. 月度棘輪重置（與 ``FuxiBacktestRunner.run`` 同一段）
2. ``broker.check_exits``：交易所託管的 TP/SL 先觸發
3. ``sm.tick(close)``：對帳（交易所為準）並推進狀態
4. 空倉時由冠軍定價 → EV 閘門 → 安全注碼 → ``sm.submit_order``
5. 把交易所回報的平倉淨損益入棘輪、關閉冠軍契約、記一筆交易
6. 收尾強平，帳本回到空倉

記帳身分與回測一致：``Σ trade.net_pnl == Δ(ratchet total_equity)``，因為
棘輪只吃交易所回報的已實現淨損益（含進出場手續費），不另計一份。

與回測路徑的差異（誠實記錄）
----------------------------
* 出場由交易所判定：``check_exits`` 用收盤價比 TP/SL，不是盤中 high/low。
  回測對同根雙觸採掃損，紙上單只看收盤，因此紙上單的觸發較寬鬆。
* ``PaperBroker`` 不計資金費，帳本沒有資金費腿。資金費仍進 EV 閘門當成本，
  使進場門檻偏保守而不會高估邊際。
* 若進場後第一根就 TP/SL，該根 ``tick`` 仍看到 ``PENDING_ORDER``（交易所
  已無倉），狀態機會在 ``max_unfilled_bars`` 根內自行取消回 IDLE；這段期間
  不開新倉。
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fuxi.backtest import FuxiBacktestConfig  # noqa: E402
from fuxi.live_state import LiveState, PositionStateMachine  # noqa: E402
from fuxi.ratchet import AsymmetricCompoundingRatchet  # noqa: E402
from fuxi.risk import (  # noqa: E402
    MIN_EV_THRESHOLD,
    PerpetualRiskAndSmoothingEngine,
)
from fuxi.rolling import RollingClosedLoopController, TrainedModel  # noqa: E402

EPS = 1e-8


@dataclass
class PaperLoopResult:
    """一整段紙上單運行的結果。"""

    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    events: List[Dict[str, Any]] = field(default_factory=list)
    n_trades: int = 0
    total_net_pnl: float = 0.0
    total_equity: float = 0.0
    active_pool: float = 0.0
    reserve_vault: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    final_state: str = LiveState.IDLE.value
    machine: Dict[str, Any] = field(default_factory=dict)

    def trade_returns(self) -> np.ndarray:
        """每筆交易對其名目本金的淨報酬（與回測結果同一口徑）。"""
        return np.array(
            [
                float(t["net_pnl_u"]) / max(float(t["margin"]) * float(t["leverage"]), EPS)
                for t in self.trades
            ],
            dtype=float,
        )

    def exit_timestamps_ms(self) -> np.ndarray:
        return np.array([int(t["exit_time_ms"]) for t in self.trades], dtype=np.int64)


def _trade_log(broker: Any) -> Optional[list]:
    """取交易所/經紀端的成交歷史清單（``PaperBroker.trades``）。

    ``PaperBrokerAdapter`` 會再包一層，所以先看內層。回傳 ``None`` 代表這個
    broker 沒有歷史清單，呼叫端改用 ``check_exits`` / ``market_close`` 的回傳值。
    """
    for cand in (getattr(broker, "broker", None), broker):
        log = getattr(cand, "trades", None)
        if isinstance(log, list):
            return log
    return None


def _as_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if is_dataclass(record):
        return asdict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(vars(record))


def run_paper_symbol(
    bars_5m: pd.DataFrame,
    features: pd.DataFrame,
    primary_signals: Optional[pd.Series] = None,
    *,
    controller: RollingClosedLoopController,
    broker: Any,
    symbol: str = "BTC",
    ratchet: Optional[AsymmetricCompoundingRatchet] = None,
    risk: Optional[PerpetualRiskAndSmoothingEngine] = None,
    config: Optional[FuxiBacktestConfig] = None,
    machine: Optional[PositionStateMachine] = None,
    model_dir: Optional[str] = None,
) -> PaperLoopResult:
    """單一 symbol、單一冠軍的逐根紙上單循環。

    ``broker`` 是狀態機 protocol 的實作（``PaperBrokerAdapter(FakeExchange)``
    或 ``PaperBrokerAdapter(PaperBroker(...))``）。``machine`` 可傳入已存在的
    狀態機以延續上一段運行；否則新建一台。
    """
    cfg = config or FuxiBacktestConfig()
    champ = controller.champion
    if champ is None and model_dir is not None:
        # 重啟語意：呼叫端（例如服務重啟後只拿到舊 champion 契約）沒有記憶體
        # 內的冠軍，但給了一個落盤目錄，就從那目錄把同一份模型載回來，而不是
        # 直接失敗。載入失敗仍維持原本的 ValueError（附上載入錯誤原文），
        # 因為「有給目錄但讀不到」是設定錯誤，不該靜默降級成空倉。
        try:
            controller.champion = TrainedModel.load(Path(model_dir))
            champ = controller.champion
        except Exception as exc:  # noqa: BLE001 - 任何載入失敗都要轉成清楚錯誤
            raise ValueError(
                "controller has no champion — train one first (Loop 2); "
                f"load from {model_dir} failed: {exc}"
            ) from exc
    if champ is None:
        raise ValueError("controller has no champion — train one first (Loop 2)")
    if primary_signals is None:
        raise ValueError("primary_signals is required (primary_dir is label-side, never a feature)")

    x = float(champ.params["x"])
    y = float(champ.params["y"])
    M = int(champ.params["M"])

    rat = ratchet or AsymmetricCompoundingRatchet(cfg.initial_capital, cfg.reinvest_ratio)
    risk_engine = risk or PerpetualRiskAndSmoothingEngine()
    # 傳入的 machine 若綁在另一個 broker 實例上，run_paper_symbol 內會把
    # exits（broker.check_exits / get_positions）記在傳入的 broker 帳上，
    # 而 fills（sm.tick / sm.submit_order）記在 machine.broker 帳上——
    # 同一筆倉位兩頭記在不同帳，會計恆等式各自成立但數字是錯的。
    # 必須是同一個實例才允許繼續；修法是把同一個 broker 傳給兩邊。
    if machine is not None and machine.broker is not broker:
        raise ValueError(
            "machine.broker is not broker: exits (broker.check_exits/get_positions) "
            "and fills (machine.tick/submit_order) would land on two different "
            "accounts. Pass the same broker instance to both "
            "(e.g. machine = PositionStateMachine(broker, symbol=symbol))."
        )
    sm = machine or PositionStateMachine(broker, symbol=symbol)

    idx = features.index.intersection(bars_5m.index)
    bars = bars_5m.loc[idx].sort_index()
    feats = features.loc[idx].sort_index()
    dirs = primary_signals.reindex(idx).fillna(0).to_numpy(dtype=int)

    closes = bars["close"].to_numpy(dtype=float)
    times_ms = (bars.index.astype("int64") // 10**6).to_numpy(dtype=np.int64)
    has_funding = "funding_rate" in bars.columns
    funding = (
        bars["funding_rate"].fillna(0.0).to_numpy(dtype=float)
        if has_funding
        else np.zeros(len(bars))
    )
    atr_all = (
        feats["tf5m_norm_atr"].fillna(0.004).to_numpy(dtype=float)
        if "tf5m_norm_atr" in feats.columns
        else np.full(len(feats), 0.004)
    )

    month_start = rat.get_status().total_equity
    month_hwm = month_start
    months = pd.DatetimeIndex(bars.index).tz_convert(None).to_period("M")
    cur_month = months[0] if len(bars) else None

    trades: List[Dict[str, Any]] = []
    equity_curve = np.empty(len(bars))
    # 進場時留下的記帳中繼資料；平倉時配對成交易紀錄。
    open_meta: Optional[Dict[str, Any]] = None
    pos_seq = 0
    log = _trade_log(broker)
    cursor = len(log) if log is not None else 0  # 接續舊帳本時不重複入帳

    def _book(fresh: List[Any], exit_ms: int) -> int:
        """把交易所回報的平倉結果入棘輪、關契約、記交易。回傳筆數。"""
        nonlocal open_meta
        booked = 0
        for raw in fresh:
            rec = _as_dict(raw)
            if not rec.get("success", True) and "net_pnl" not in rec:
                continue
            net = float(rec.get("net_pnl") or 0.0)
            meta = open_meta or {}
            pos_id = meta.get("pos_id")
            if pos_id:
                controller.close_position(pos_id)
            notional = float(meta.get("margin", 0.0)) * float(meta.get("leverage", 1.0))
            gross = float(rec.get("gross_pnl") or 0.0)
            rat.update_intraday_pnl(net)
            trades.append(
                {
                    "pos_id": pos_id,
                    "model_id": meta.get("model_id", "unknown"),
                    "direction": int(meta.get("direction", 0)),
                    "entry_price": float(rec.get("entry_price") or meta.get("entry_price") or 0.0),
                    "exit_price": float(rec.get("exit_price") or 0.0),
                    "margin": float(meta.get("margin", 0.0)),
                    "leverage": float(meta.get("leverage", 1.0)),
                    "gross_pnl": gross,
                    "fees": float(rec.get("fees") or 0.0),
                    "gross_ret": gross / max(notional, EPS),
                    "net_ret": net / max(notional, EPS),
                    "net_pnl_u": net,
                    "exit_reason": str(rec.get("exit_reason") or rec.get("reason") or ""),
                    "entry_time_ms": int(meta.get("entry_ms", 0)),
                    "exit_time_ms": int(exit_ms),
                }
            )
            open_meta = None
            booked += 1
        return booked

    def _drain(exit_ms: int) -> int:
        """從交易所成交歷史抓這根新增的平倉紀錄併入帳。"""
        nonlocal cursor
        if log is None:
            return 0
        fresh = list(log[cursor:])
        cursor = len(log)
        return _book(fresh, exit_ms)

    def _settle_flat() -> None:
        rat.settle_rolling_period(is_flat=True)

    def _has_position(mark: float) -> bool:
        positions = broker.get_positions({symbol: float(mark)}) or []
        return any(p.get("symbol") == symbol for p in positions)

    for i in range(len(bars)):
        # ---- 1. 月度棘輪重置 ---- #
        bar_month = months[i]
        if bar_month != cur_month:
            cur_month = bar_month
            status = rat.get_status()
            month_start = status.total_equity
            month_hwm = month_start
        else:
            month_hwm = max(month_hwm, rat.get_status().total_equity)

        # ---- 2. 交易所託管的 TP/SL 先觸發 ---- #
        exits = broker.check_exits({symbol: float(closes[i])}) or []
        if log is None:
            _book(list(exits), int(times_ms[i]))

        # ---- 3. 對帳 + 狀態推進（到期平倉在此發生） ---- #
        n_events_before = len(sm.events)
        sm.tick(float(closes[i]))
        if log is None:
            # tick 內到期平倉的 market_close 回傳寫在 position_closed 事件裡；
            # 這裡從事件尾撈出來走同一條 _book。TP/SL 已在步驟 2 記過，
            # 以 open_meta 是否已清來防重記（_book 記完即清）。
            for ev in sm.events[n_events_before:]:
                if ev.get("event") == "position_closed" and isinstance(ev.get("close"), dict):
                    _book([ev["close"]], int(times_ms[i]))
        # ---- 4. 空倉才定價進場 ---- #
        if sm.state is LiveState.IDLE and dirs[i] != 0 and not controller.pass_regime:
            row = feats.iloc[i : i + 1]
            p_cal = float(np.asarray(champ.calibrated_proba(row)).reshape(-1)[0])
            if p_cal > cfg.entry_threshold:
                status = rat.get_status()
                decision = risk_engine.compute_safe_sizing(
                    prob_cal=p_cal,
                    direction=int(dirs[i]),
                    x=x,
                    y=y,
                    target_leverage=cfg.target_leverage,
                    ratcheted_base_capital=status.kelly_base_capital,
                    total_equity=status.total_equity,
                    month_start_equity=month_start,
                    month_hwm_equity=month_hwm,
                    rolling_penalty_scalar=controller.kelly_penalty_scalar,
                    current_norm_atr=float(atr_all[i]),
                    funding_rate=float(funding[i]) if has_funding else 0.0,
                )
                if decision.approved and decision.ev > MIN_EV_THRESHOLD:
                    pos_seq += 1
                    pos_id = f"PX{pos_seq:05d}"
                    submitted = sm.submit_order(
                        int(dirs[i]),
                        float(closes[i]),
                        notional=float(decision.margin) * float(decision.leverage),
                        leverage=float(decision.leverage),
                        model_id=champ.model_id,
                        x=x,
                        y=y,
                        max_hold_bars=M,
                    )
                    if submitted and sm.contract is not None:
                        contract = controller.register_new_position(
                            pos_id,
                            int(dirs[i]),
                            float(sm.contract.entry_price),
                            float(decision.margin),
                            float(decision.leverage),
                            i,
                        )
                        open_meta = {
                            "pos_id": pos_id,
                            "model_id": contract.model_id,
                            "direction": int(dirs[i]),
                            "entry_price": float(contract.entry_price),
                            "margin": float(decision.margin),
                            "leverage": float(decision.leverage),
                            "entry_ms": int(times_ms[i]),
                        }

        # ---- 5. 平倉入帳（含到期／取消平倉） ---- #
        booked = 0
        if log is not None:
            booked = _drain(int(times_ms[i]))
        if booked and cfg.settle_every_bars and (i % cfg.settle_every_bars == 0):
            _settle_flat()

        equity_curve[i] = rat.get_status().total_equity

    # ---- 6. 收尾強平（含到期會計） ---- #
    last_mark = float(closes[-1]) if len(bars) else 0.0
    if len(bars):
        # 先推一根 reconcile tick：若 broker 已在 exit/market_close 把倉平掉，
        # tick 會把狀態機同步到 COOLDOWN/IDLE，後面就不用再平一次。
        sm.tick(last_mark)
    if len(bars) and _has_position(last_mark) and sm.contract is not None:
        closed = broker.market_close(symbol, last_mark, "eod_flatten")
        if log is None:
            _book([closed] if closed else [], int(times_ms[-1]))
        else:
            _drain(int(times_ms[-1]))
        equity_curve[-1] = rat.get_status().total_equity
        sm.tick(last_mark)  # 把平倉同步回狀態機，帳本回到空倉
    elif log is not None and len(bars) and open_meta is not None:
        # 交易所已無倉（TP/SL/到期觸發），但該筆還沒進帳：補抓。
        _drain(int(times_ms[-1]))
        equity_curve[-1] = rat.get_status().total_equity
        sm.tick(last_mark)

    _settle_flat()
    final = rat.get_status()
    curve = pd.Series(equity_curve, index=bars.index)
    peak = curve.cummax()
    dd = float(((peak - curve) / np.maximum(peak, EPS)).max()) if len(curve) else 0.0
    wins = sum(1 for t in trades if t["net_pnl_u"] > 0)
    return PaperLoopResult(
        trades=trades,
        equity_curve=curve,
        events=list(sm.events),
        n_trades=len(trades),
        total_net_pnl=float(sum(t["net_pnl_u"] for t in trades)),
        total_equity=final.total_equity,
        active_pool=final.active_pool,
        reserve_vault=final.reserve_vault,
        win_rate=(wins / len(trades)) if trades else 0.0,
        max_drawdown=dd,
        final_state=sm.state.value,
        machine=sm.to_dict(),
    )

"""Training agent loop：兩種搜索模式

Mode A (SPEC)：LLM 直接輸出 StrategySpec JSON（原有）
Mode B (HYPER_AI)：LLM 輸出自然語言請求 → Hyper AI pipeline (Prompt AI → Program AI) → StrategySpec → 評估

與 strategy_spec 的關係
----------------------
- 只依賴 backend/backtest/strategy_spec.py 的既有接縫
  （StrategySpec.from_dict / validate_spec / evaluate_spec_individual），
  不重寫任何評估或聚合邏輯。
- LLM 呼叫走 services/ai_decision_service 的 build_llm_payload /
  build_llm_headers / detect_api_format，設定來源是
  services/hyper_ai_service.get_llm_config（provider=custom、model=auto-free）。

流程（run_training_loop）
------------------------
Mode A：build_prompt -> _call_llm -> parse_candidate -> evaluate_spec_individual
Mode B：build_prompt_hyper -> _call_llm -> extract_request -> run_hyper_ai_offline -> evaluate_spec_individual
本模組為新增檔案，不修改任何既有檔案。
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from backtest.strategy_spec import (
        StrategySpec,
        evaluate_spec_individual,
        validate_spec,
    )
except ImportError:  # pragma: no cover - 非標準路徑下的退路
    from backend.backtest.strategy_spec import (  # type: ignore
        StrategySpec,
        evaluate_spec_individual,
        validate_spec,
    )

try:
    import requests
except ImportError:  # pragma: no cover - 無 requests 環境（僅單元測試用）
    requests = None  # type: ignore[assignment]

try:
    from services.ai_decision_service import (
        build_llm_headers,
        build_llm_payload,
        detect_api_format,
    )
    from services.hyper_ai_service import get_llm_config
except ImportError:  # pragma: no cover - 非標準路徑下的退路
    from backend.services.ai_decision_service import (  # type: ignore
        build_llm_headers,
        build_llm_payload,
        detect_api_format,
    )
    from backend.services.hyper_ai_service import get_llm_config  # type: ignore

try:
    from backtest.hyper_ai_pipeline import (
        run_hyper_ai_offline,
        strategy_code_to_spec,
        evaluate_hyper_ai_output,
    )
except ImportError:  # pragma: no cover
    from backend.backtest.hyper_ai_pipeline import (  # type: ignore
        run_hyper_ai_offline,
        strategy_code_to_spec,
        evaluate_hyper_ai_output,
    )
 
# _call_llm 的簽名固定（易於測試 mock），故 db session 工廠以模組狀態傳遞；
# run_training_loop 進入時設定、離開時還原。
_DB_FACTORY: Optional[Any] = None


@contextmanager
def _session(db_factory: Optional[Any]) -> Iterator[Any]:
    """取 db session：顯式 db_factory 優先，否則函式內延遲匯入 SessionLocal。

    呼叫方負責的 with 區塊離開時一律 close。
    """
    factory = db_factory if db_factory is not None else _DB_FACTORY
    if factory is None:
        try:
            from database.connection import SessionLocal
        except ImportError:
            from backend.database.connection import SessionLocal  # type: ignore
        factory = SessionLocal
    db = factory()
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            pass


# ========== 提示詞 ==========

SYSTEM_PROMPT_TEMPLATE = """You are a systematic trading strategy designer. \
Propose improved StrategySpec structures for a crypto backtest engine.

A StrategySpec is JSON:
{{"filters": [{{"kind": "...", "params": {{...}}}}, ...],
  "risk": {{"tp_pct": ..., "sl_pct": ..., "portion": ..., "leverage": ...}},
  "logic": "all_pass"}}

Allowed filter kinds and params:
- ema_cross: ema_fast [10,30], ema_slow [40,80], ema_tol_buy [0.995,0.9995], ema_tol_sell [1.0005,1.002]
- donchian_breakout: lookback [41,81], threshold [0.97,0.995]
- rsi_band: low [25,45], high [55,75]
- ict_confirm: ote_threshold [0.0,0.75], require_alignment [0,1], require_bull_structure [0,1], require_discount [0,1], swing_len [1,6], weight [0.0,1.0]

Risk params: tp_pct [0.04,0.12], sl_pct [0.02,0.08], portion [0.15,0.5], leverage [1,3]. max_positions is fixed at 3.
logic must be exactly "all_pass" (declared filters all pass together).

Goal: maximize fitness on the same backtest windows as the baseline.
Fitness reality (work within it, do not fight it):
- Trade sweet spot is 60-150 per year. Below 60 is penalized linearly
  (sample too small); above 150 is penalized quadratically (cost erosion).
  A very low trade count only survives when MaxDD is tiny, so never raise
  trade count unless MaxDD stays under 8%.
- MaxDD above 8% is penalized heavily. Keep portion x leverage small enough
  that an adverse move stays inside budget.
- Each added filter ANDs another gate and usually removes trades. Prefer
  tuning the current structure over adding filters.
Hard rules:
1. Output ONE valid JSON object only — no explanation, no markdown outside the JSON.
2. Every param within the ranges above.
3. Between 1 and 4 filters (max 4).
4. Change something meaningful vs. the baseline (structure or tuning).
5. Keep the trade count within +/-50% of the baseline unless you also cut
   portion and leverage to pay for the extra turnover.
"""

USER_PROMPT_TEMPLATE = """Baseline spec (fitness {fitness}, gate={gate}, pnl={pnl}%, \
maxdd={maxdd}%, sharpe={sharpe}, pf={pf}, trades={trades}):

{baseline_json}

{history_block}

Propose ONE improved spec as a single JSON object."""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
    return "N/A" if value is None else str(value)


def build_prompt(baseline_spec_dict: Dict[str, Any],
                 eval_summary: Dict[str, Any],
                 round_no: int,
                 history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """組 auto-free 的訊息：system（Spec 形狀 + 硬規則）+ user（baseline、
    指標、前輪歷史、要求回一個改進 spec）。保持精簡。"""
    baseline = deepcopy(baseline_spec_dict)
    history_rows = []
    for h in (history or []):
        history_rows.append(
            "round {r}: spec_id={sid} delta_fitness={df} changed={ch}".format(
                r=h.get("round", "?"),
                sid=h.get("spec_id", "n/a"),
                df=h.get("delta_fitness", "n/a"),
                ch=h.get("changed", "n/a"),
            ))
    history_block = ""
    if history_rows:
        history_block = ("Previous rounds:\n" + "\n".join(history_rows)
                         + "\n\nAvoid repeating rejected/edited ideas; "
                           "try a different direction.")
    user = USER_PROMPT_TEMPLATE.format(
        fitness=_fmt(eval_summary.get("fitness")),
        gate=str(bool(eval_summary.get("gate_annual"))).lower(),
        pnl=_fmt(eval_summary.get("pnl_365d")),
        maxdd=_fmt(eval_summary.get("maxdd_365d")),
        sharpe=_fmt(eval_summary.get("sharpe_365d")),
        pf=_fmt(eval_summary.get("pf")),
        trades=_fmt(eval_summary.get("trades_365d")),
        baseline_json=json.dumps(baseline, sort_keys=True, ensure_ascii=False),
        history_block=history_block,
    ) + f"\nRound {round_no}."
    return [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": user}]


# ========== Hyper AI Mode Prompts ==========

HYPER_AI_SYSTEM_PROMPT = """You are a systematic trading strategy designer.
Your job: output a NATURAL LANGUAGE REQUEST for the Hyper AI trading team
(Prompt AI → Program AI, sandboxed Python strategy code).

Your output must be a single JSON object:
{"request": "your natural language request here"}

DROP the old family: after 25 rounds the EMA-cross / Donchian-breakout /
RSI-band / ICT_OTE combination has no positive edge (all fitness negative).
Design entries from the NEW indicator family below. Never use EMA cross,
Donchian, RSI band, or ICT as the primary entry gate.

New indicator family (exact get_indicator(symbol, NAME, "1h") shapes —
copy these dict keys verbatim into the request so Program AI emits
working access code):
- MACD: data.get_indicator(symbol, "MACD", "1h") -> {"macd": float,
  "signal": float, "histogram": float}. Bullish: histogram > 0 and
  macd > signal. Bearish: histogram < 0 and macd < signal.
- BOLL: data.get_indicator(symbol, "BOLL", "1h") -> {"upper": float,
  "middle": float, "lower": float}. Long only when price <= lower
  (bounce) or price >= upper (squeeze breakout) — pick ONE per side.
- STOCH: data.get_indicator(symbol, "STOCH", "1h") -> {"k": float,
  "d": float}. Oversold k < 20 with k > d for longs; overbought
  k > 80 with k < d for shorts. STOCH oscillates 0-100.
- VWAP: data.get_indicator(symbol, "VWAP", "1h") -> {"value": float}.
  Long bias when price > vwap, short bias when price < vwap.
- OBV: data.get_indicator(symbol, "OBV", "1h") -> {"value": float}.
  OBV is cumulative volume and the Strategy object is rebuilt EVERY
  trigger, so NO slope/history is available (self.* never persists).
  Only use its SIGN as a weak same-trigger filter (obv > 0 for longs,
  obv < 0 for shorts). Never compare to a fixed level.
- ATR14: data.get_indicator(symbol, "ATR14", "1h") -> {"value": float}
  in price units. Use for volatility stops: sl_dist = 1.5*ATR,
  tp_dist = 3.0*ATR. ATR entries must ALSO gate on a directional
  signal (MACD histogram sign or VWAP side) — ATR alone has no direction.

Semantic pairing (mixing opposite regimes yields ~0 trades):
- OVERSOLD family (use together): STOCH k<20 with k>d, price <= BOLL
  lower (bounce), price < VWAP about to reclaim. All three say
  "washed-out long".
- OVERBOUGHT family (use together): STOCH k>80 with k<d, price >= BOLL
  upper (breakout), price < VWAP (short continuation). All three say
  "weak short".
- TREND family (use together): MACD histogram>0, price > VWAP, price
  near/above BOLL middle.
- NEVER combine STOCH k<20 (oversold) with price >= BOLL upper
  (overbought breakout) on the same side — they almost never co-occur
  and the pair yields ~0 trades (v21 r5 proved this exact failure).
- NEVER combine STOCH k>80 (overbought) with price <= BOLL lower.

Entry recipe (state-based, NEVER self.prev_*/crossed/self.* event gates —
should_trade fires per scheduled trigger on a FRESH object, not per bar):
- Pick ONE family per side (oversold-long + overbought-short, OR
  trend both-sides) and 2 signals max per side.
- Add a 3rd filter (ATR stop) only if trades > 150/yr.
- Target 60-150 trades/year. MaxDD > 8% is heavily penalized, so keep
  portion 0.2-0.35 and leverage 1-2.

Hard rules:
1. Output ONE valid JSON object only: {"request": "..."}
2. Request names exact indicator keys above, with numeric thresholds
  (e.g. "STOCH k<20 with k>d", "MACD histogram>0", "price within 0.3%
  of BOLL lower").
3. Request must be state-comparison only (price vs level, k vs d,
  macd vs signal) — forbid crossover-event phrasing.
4. Change entry family vs. the baseline; tune thresholds, not structure.
5. Keep it concise (2-4 sentences)."""


HYPER_AI_USER_PROMPT = """Best fitness so far: {fitness} (gate={gate}, pnl={pnl}%,
maxdd={maxdd}%, sharpe={sharpe}, pf={pf}, trades={trades}).
The old EMA/Donchian/RSI/ICT family is retired — do NOT reference it.

{history_block}

Propose ONE improved natural language request for the Hyper AI team as JSON: {{"request": "..."}} — entry must come from the NEW indicator family in the system prompt."""


def build_prompt_hyper(baseline_spec_dict: Dict[str, Any],
                       eval_summary: Dict[str, Any],
                       round_no: int,
                       history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build prompt for Hyper AI mode: LLM outputs natural language request.

    刻意不把 baseline spec JSON 貼進 prompt：baseline 是 genome_to_spec 產物，
    形狀全是 ema_cross/donchian/rsi_band/ict_confirm。舊家族在 pinned 窗口
    25 輪全負 fitness，貼 JSON 只會把 LLM 錨回同一組欄位名。改用「指標家族」
    描述（HYPER_AI_SYSTEM_PROMPT）加每輪實測數字驅動。
    """
    history_rows = []
    for h in (history or []):
        m = h.get("metrics") or {}
        # 訓練迴圈本來只給 LLM 看 delta_fitness，LLM 完全不知道上一輪
        # 是「交易太多淹死」（r2 952 筆、r3 3448 筆）還是「交易太少餓死」
        # （r4 3 筆）。v17 起把每輪的 trades/pnl/dd/pf 一起寫進 history，
        # LLM 才能朝「60-150 筆、dd<8%、pnl>0」的正確方向修正。
        history_rows.append(
            "round {r}: delta_fitness={df} trades={t} pnl={p}% maxdd={dd}% "
            "pf={pf} req={req}".format(
                r=h.get("round", "?"),
                df=h.get("delta_fitness", "n/a"),
                t=m.get("trades_365d", "?"),
                p=m.get("pnl_365d", "?"),
                dd=m.get("maxdd_365d", "?"),
                pf=m.get("pf", "?"),
                req=str((h.get("changed") or {}).get("request", "?"))[:120],
            ))
    history_block = ""
    seed = (eval_summary or {}).get("seed_structure") if round_no == 1 else None
    if seed:
        # 種子輪（v24 起）：同一份 STOCH/BOLL 代碼同時跑 BTC+ETH+SOL。
        # 前提：v22 r3 的 19 筆 +4.74% 是該結構在單幣 180d 的方向。
        # 跑三幣合計頻率（目標 60-150 筆/年），保持同一 entry 邏輯不變。
        # 若三幣同為負：結構無 alpha，不是頻率問題。
        family = seed.get("family") or "STOCH/BOLL"
        history_block = (
            f"SEED STRUCTURE ({family}): {seed['structure']}\n"
            f"Single-coin baseline: {seed['trades']} trades/yr, "
            f"{seed['pnl']:+.2f}% PnL, PF {seed['pf']:.3f}.\n"
            "Run the SAME entry logic on all symbols (BTC+ETH+SOL). "
            "Do NOT change thresholds or add filters — this run measures "
            "whether frequency scales with symbols, not whether tuning helps."
        )
    elif history_rows:
        history_block = ("Previous rounds:\n" + "\n".join(history_rows)
                         + "\n\nAvoid repeating rejected/edited ideas; "
                           "try a different indicator pair.")
    user = HYPER_AI_USER_PROMPT.format(
        fitness=_fmt(eval_summary.get("fitness")),
        gate=str(bool(eval_summary.get("gate_annual"))).lower(),
        pnl=_fmt(eval_summary.get("pnl_365d")),
        maxdd=_fmt(eval_summary.get("maxdd_365d")),
        sharpe=_fmt(eval_summary.get("sharpe_365d")),
        pf=_fmt(eval_summary.get("pf")),
        trades=_fmt(eval_summary.get("trades_365d")),
        history_block=history_block,
    ) + f"\nRound {round_no}."
    return [{"role": "system", "content": HYPER_AI_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ========== Hyper AI Mode Evaluation ==========

def _run_hyper_ai_eval(
    request_text: str,
    symbols: List[str],
    exchange: str,
    backtest_days: int,
    end_time_ms: Optional[int],
    kline_period: str,
    generation: int,
    total_generations: int,
    db_factory: Optional[Any],
) -> Dict[str, Any]:
    """Run Hyper AI pipeline and evaluate the resulting code DIRECTLY."""
    try:
        from backtest.hyper_ai_pipeline import (
            run_hyper_ai_offline,
            evaluate_strategy_code,
        )
    except ImportError:
        from backend.backtest.hyper_ai_pipeline import (  # type: ignore
            run_hyper_ai_offline,
            evaluate_strategy_code,
        )
    try:
        # Stage 1: LLM request -> Hyper AI pipeline -> raw strategy code
        strategy_code, trace = run_hyper_ai_offline(
            user_request=request_text,
            exchange=exchange,
            symbols=symbols,
            db_factory=db_factory,
        )
        if not strategy_code or "class Strategy" not in strategy_code:
            # 抽取器已拒收散文/規格要點，這裡補一道：沒有 Strategy 類別就
            # 不進回測。空 code 會得到 0 筆交易的假 fitness（-28 而非 -1e9），
            # 訓練迴圈會誤以為「這輪有效但很差」，浪費一輪還污染 best 追蹤。
            raise RuntimeError(
                f"Program AI returned no Strategy code (len={len(strategy_code or '')})"
            )
        # A（v18, v20 擴大）：靜態預檢事件型入口與跨觸發實例狀態。
        # 沙箱每次 trigger 都重建 Strategy 實例（engine 兩條 _execute 路徑皆然），
        # 所以 self.prev_*/crossed（v15 實證 ~1 筆/180d）與 self.obv_history/
        # self.last_obv（v20 r2 實證：邏輯退化為常數路徑）都注定無效。
        # 直接記 round_invalid，省下 200s 回測。
        _lc = strategy_code.lower()
        _has_state_abuse = (
            "prev_" in _lc or "crossed" in _lc
            or "self.obv" in _lc or "self.last_" in _lc
            or "self.history" in _lc or "self.prices" in _lc
        )
        if _has_state_abuse and "get_indicator" in _lc:
            raise RuntimeError(
                "static_reject: instance-state entry (self.prev_*/crossed/obv/last_/history/prices). "
                "Strategy is rebuilt every trigger — self.* never persists. "
                "Use same-trigger state comparison instead."
            )
        metrics = evaluate_strategy_code(
            strategy_code,
            symbols=symbols,
            exchange=exchange,
            generation=generation,
            total_generations=total_generations,
            backtest_days=backtest_days,
            end_time_ms=end_time_ms,
            kline_period=kline_period,
            db_factory=db_factory,
        )
        metrics["strategy_code"] = strategy_code
        metrics["spec_id"] = trace.get("stages", {}).get("program_ai", {}).get("trace", {}).get("message_id", f"hyper_{generation}")
        metrics["pipeline_trace"] = trace
        return metrics
    except Exception as e:
        # Return a failed evaluation dict
        return {
            "fitness": -1e9,
            "gate_annual": False,
            "pnl_365d": -100.0,
            "maxdd_365d": 100.0,
            "sharpe_365d": -10.0,
            "pf": 0.0,
            "pf_ok": False,
            "trades_365d": 0,
            "win_rate": 0.0,
            "spec_id": "error",
            "error": str(e),
        }


def _extract_request_from_llm(text: str) -> Tuple[Optional[str], List[str]]:
    """Extract the request string from LLM output JSON."""
    errors = []
    text = _strip_fences(text)
    try:
        data = json.loads(text)
        request = data.get("request")
        if not request or not isinstance(request, str):
            errors.append("Missing or invalid 'request' field in JSON")
            return None, errors
        if len(request) < 10:
            errors.append("Request too short (< 10 chars)")
            return None, errors
        return request.strip(), []
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return None, errors

# ========== 解析 ==========

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """去 markdown 圍欄；沒有圍欄時返回原文。"""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def _extract_json_object(text: str) -> str:
    """取第一個 '{' 到最後一個 '}'，吃掉 JSON 前後的散文。"""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_candidate(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """從 LLM 輸出抽 JSON -> StrategySpec.from_dict -> validate_spec。

    回傳 (spec_dict 或 None, 錯誤字串清單)。對任何壞輸入不拋例外。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None, ["empty LLM response"]
    body = _extract_json_object(_strip_fences(cleaned).strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return None, [f"response is not valid JSON: {e}"]
    if not isinstance(data, dict):
        return None, ["response JSON is not an object"]
    try:
        spec = StrategySpec.from_dict(data)
    except (KeyError, TypeError, ValueError) as e:
        return None, [f"spec shape invalid: {e}"]
    spec_errors = validate_spec(spec)
    if len(spec.filters) > 4:
        spec_errors.append(f"filters={len(spec.filters)} 超過上限 4")
    if spec_errors:
        return None, spec_errors
    return spec.to_dict(), []


# ========== LLM 呼叫 ==========

def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 2000) -> str:
    """單純的一次 LLM 呼叫，回傳文字內容；呼叫失敗拋例外，由 caller 處理。"""
    if requests is None:
        raise RuntimeError("requests not available")
    with _session(None) as db:
        cfg = get_llm_config(db)
    if not cfg.get("configured"):
        raise RuntimeError("LLM 未配置（get_llm_config -> configured False）")
    endpoint, api_format = detect_api_format(cfg["base_url"])
    if api_format is None:
        api_format = cfg.get("api_format") or "openai"
        endpoint = cfg["base_url"].rstrip("/") + (
            "/v1/messages" if api_format == "anthropic" else "/chat/completions")
    payload = build_llm_payload(model=cfg["model"], messages=messages,
                                api_format=api_format,
                                max_tokens=max_tokens, temperature=0.7)
    headers = build_llm_headers(api_format, cfg["api_key"], cfg["base_url"])
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if api_format == "anthropic":
        content = data.get("content") or []
        return str(content[0].get("text", "")) if content else ""
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", ""))


# ========== 迴圈 ==========

def _run_evaluate(spec_dict: Dict[str, Any], symbols: List[str], **kw: Any) -> Dict[str, Any]:
    """評估接縫的間接層，方便測試 monkeypatch（training_agent._run_evaluate）。"""
    return evaluate_spec_individual(spec_dict, symbols, **kw)


def run_training_loop(baseline_spec_dict: Dict[str, Any],
                      symbols: List[str],
                      rounds: int = 3,
                      backtest_days: int = 60,
                      end_time_ms: Optional[int] = None,
                      kline_period: str = "1h",
                      exchange: str = "hyperliquid",
                      db_factory: Optional[Any] = None) -> Dict[str, Any]:
    """訓練迴圈：LLM 提案 -> 驗證 -> 評估 -> 保留最佳。

    db_factory：回傳 Session 的 callable；None 時 _session 在函式內延遲匯入
    database.connection.SessionLocal（避免匯入本模組即建立引擎）。
    """
    global _DB_FACTORY
    prev_factory = _DB_FACTORY
    _DB_FACTORY = db_factory
    try:
        return _run_training_loop(baseline_spec_dict, symbols, rounds=rounds,
                                  backtest_days=backtest_days,
                                  end_time_ms=end_time_ms,
                                  kline_period=kline_period,
                                  exchange=exchange)
    finally:
        _DB_FACTORY = prev_factory


def _run_training_loop(baseline_spec_dict: Dict[str, Any],
                       symbols: List[str],
                       rounds: int = 3,
                       backtest_days: int = 60,
                       end_time_ms: Optional[int] = None,
                       kline_period: str = "1h",
                       exchange: str = "hyperliquid") -> Dict[str, Any]:
    eval_kwargs = dict(exchange=exchange, backtest_days=backtest_days,
                       end_time_ms=end_time_ms, kline_period=kline_period)

    def _spec_eval(spec_dict: Dict[str, Any], generation: int) -> Dict[str, Any]:
        return _run_evaluate(deepcopy(spec_dict), list(symbols), generation=generation,
                             total_generations=max(rounds, 1), **eval_kwargs)

    baseline_metrics = _spec_eval(baseline_spec_dict, generation=1)
    best_spec = deepcopy(baseline_spec_dict)
    best_metrics = baseline_metrics
    history: List[Dict[str, Any]] = []

    for round_no in range(1, max(1, rounds) + 1):
        messages = build_prompt(deepcopy(best_spec), best_metrics, round_no, history)
        spec_dict: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        for attempt in range(2):
            if attempt == 1 and error:
                messages = build_prompt(deepcopy(best_spec), best_metrics,
                                        round_no, history)
                messages[1]["content"] += (
                    f"\nYour previous attempt was invalid: {error}. "
                    "Return one valid JSON spec.")
            try:
                text = _call_llm(messages)
            except Exception as e:
                error = f"llm_call_failed: {e}"
                break
            parsed, p_errors = parse_candidate(text)
            if parsed is not None:
                spec_dict = parsed
                error = None
                break
            error = "; ".join(p_errors)
        if spec_dict is None:
            history.append({"round": round_no, "spec_id": None, "fitness": None,
                            "accepted": False, "error": error})
            continue
        candidate = StrategySpec.from_dict(spec_dict)
        metrics = _spec_eval(candidate.to_dict(), generation=round_no + 1)
        fitness = float(metrics.get("fitness", -1000.0) or -1000.0)
        best_fitness = float(best_metrics.get("fitness", -1000.0) or -1000.0)
        accepted = fitness > best_fitness
        history.append({
            "round": round_no,
            "spec_id": candidate.spec_id(),
            "fitness": fitness,
            "accepted": accepted,
            "error": None,
        })
        if accepted:
            best_spec = deepcopy(spec_dict)
            best_metrics = metrics
            next_hist = dict(history[-1])
            next_hist["delta_fitness"] = round(fitness - best_fitness, 6)
            next_hist["changed"] = deepcopy(spec_dict)
            history[-1] = next_hist
        else:
            next_hist = dict(history[-1])
            next_hist["delta_fitness"] = round(fitness - best_fitness, 6)
            next_hist["changed"] = deepcopy(spec_dict)
            history[-1] = next_hist

    return {
        "best_spec": deepcopy(best_spec),
        "best_metrics": deepcopy(best_metrics),
        "rounds": history,
        "baseline_fitness": baseline_metrics.get("fitness"),
    }


def run_training_loop_hyper(baseline_spec_dict: Dict[str, Any],
                            symbols: List[str],
                            rounds: int = 3,
                            backtest_days: int = 60,
                            end_time_ms: Optional[int] = None,
                            kline_period: str = "1h",
                            exchange: str = "hyperliquid",
                            db_factory: Optional[Any] = None,
                            progress_cb: Optional[Any] = None,
                            round_timeout_sec: float = 0,
                            history_cb: Optional[Any] = None,
                            seed_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hyper AI 模式訓練迴圈：LLM 輸出自然語言請求 -> Hyper AI pipeline -> 評估 -> 保留最佳。

    與 SPEC 模式共用同一套評估標準 (evaluate_spec_individual)，只換「提案生成方式」。
    progress_cb(round_no, status)：長時間背景任務落盤進度（避免被殺後無從得知卡在哪）。
    round_timeout_sec：單輪超時（秒）；0 = 不限制。Program AI 串流可能永不返回，
    超時該輪作廢繼續下一輪，daemon 執行緒泄漏可接受（程序級上限）。
    history_cb(history, best_metrics)：每輪落盤快照。
    seed_structure：已知正 PnL 但頻率不足的結構（dict: structure/trades/pnl/pf），
    只在第 1 輪寫進 prompt 當放寬起點；之後輪次看實測數字。
    """
    global _DB_FACTORY
    prev_factory = _DB_FACTORY
    _DB_FACTORY = db_factory
    try:
        return _run_training_loop_hyper(baseline_spec_dict, symbols, rounds=rounds,
                                        backtest_days=backtest_days,
                                        end_time_ms=end_time_ms,
                                        kline_period=kline_period,
                                        exchange=exchange,
                                        progress_cb=progress_cb,
                                        round_timeout_sec=round_timeout_sec,
                                        history_cb=history_cb,
                                        seed_structure=seed_structure)
    finally:
        _DB_FACTORY = prev_factory


def _run_training_loop_hyper(baseline_spec_dict: Dict[str, Any],
                             symbols: List[str],
                             rounds: int = 3,
                             backtest_days: int = 60,
                             end_time_ms: Optional[int] = None,
                             kline_period: str = "1h",
                             exchange: str = "hyperliquid",
                             progress_cb: Optional[Any] = None,
                             round_timeout_sec: float = 0,
                             history_cb: Optional[Any] = None,
                             seed_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """round_timeout_sec > 0 時，每輪在背景執行緒跑，超時即作廢該輪繼續下一輪。

    Program AI 的 HTTP 串流可能永遠不返回（v5 r2/r5 各卡 20+ 分鐘，socket 停在
    CLOSE_WAIT）。沒有超時就整份訓練卡死；沒有 history_cb 就一掛全丟。
    """
    def _cb(round_no: int, status: str) -> None:
        try:
            if progress_cb is not None:
                progress_cb(round_no, status)
        except Exception:
            pass
    eval_kwargs = dict(exchange=exchange, backtest_days=backtest_days,
                       end_time_ms=end_time_ms, kline_period=kline_period)

    def _hyper_eval(request_text: str, generation: int) -> Dict[str, Any]:
        return _run_hyper_ai_eval(request_text, list(symbols),
                                  generation=generation,
                                  total_generations=max(rounds, 1),
                                  db_factory=_DB_FACTORY,
                                  **eval_kwargs)

    # Evaluate baseline using SPEC path (convert baseline spec to request first)
    # For baseline, we just use the spec directly via SPEC eval
    baseline_metrics = _run_evaluate(deepcopy(baseline_spec_dict), list(symbols),
                                     generation=1, total_generations=max(rounds, 1),
                                     **eval_kwargs)
    best_spec = deepcopy(baseline_spec_dict)
    best_metrics = baseline_metrics
    history: List[Dict[str, Any]] = []

    for round_no in range(1, max(1, rounds) + 1):
        _cb(round_no, "build_prompt")
        # 種子結構只在第 1 輪注入；之後輪次改用實測 history 驅動。
        _eval_for_prompt = dict(best_metrics or {})
        if round_no == 1 and seed_structure:
            _eval_for_prompt["seed_structure"] = seed_structure
        messages = build_prompt_hyper(deepcopy(best_spec), _eval_for_prompt,
                                      round_no, history)
        request_text: Optional[str] = None
        error: Optional[str] = None
        for attempt in range(2):
            if attempt == 1 and error:
                messages = build_prompt_hyper(deepcopy(best_spec), best_metrics,
                                              round_no, history)
                messages[1]["content"] += (
                    f"\nYour previous attempt was invalid: {error}. "
                    "Return one valid JSON with 'request' field.")
            try:
                _cb(round_no, f"llm_call_attempt{attempt}")
                text = _call_llm(messages)
                _cb(round_no, f"llm_done_attempt{attempt}")
            except Exception as e:
                error = f"llm_call_failed: {e}"
                _cb(round_no, f"llm_failed: {error[:80]}")
                break
            request_text, p_errors = _extract_request_from_llm(text)
            if request_text is not None:
                error = None
                break
            error = "; ".join(p_errors)
        if request_text is None:
            history.append({"round": round_no, "spec_id": None, "fitness": None,
                            "accepted": False, "error": error,
                            "request": None})
            _cb(round_no, f"no_request: {(error or '')[:80]}")
            continue
        _cb(round_no, f"pipeline_start: {(request_text or '')[:60]}")

        def _round_work(rt=request_text, rn=round_no):
            metrics = _hyper_eval(rt, generation=rn + 1)
            fitness = float(metrics.get("fitness", -1e9) or -1e9)
            accepted = fitness > float(best_metrics.get("fitness", -1e9) or -1e9)
            code = metrics.get("strategy_code") or ""
            rec = {
                "round": rn,
                "spec_id": metrics.get("spec_id"),
                "fitness": fitness,
                "accepted": accepted,
                "error": metrics.get("error"),
                "request": rt,
                # 診斷：讓事後能看出每輪到底產出什麼、為什麼 fitness 相同。
                # v22 前快照只存 head 前 200 字 —— 正收益輪的完整代碼全部
                # 不可復用（35 輪訓練等於白跑）。v22 起存全文，promote 的
                # 種子直接從快照取 code 再評估。
                "strategy_code": code,
                "strategy_code_len": len(code),
                "strategy_code_head": (code[:200] if code else ""),
                "risk": metrics.get("risk"),
                "metrics": {k: metrics.get(k) for k in
                            ("gate_annual", "pnl_365d", "maxdd_365d", "sharpe_365d",
                             "pf", "trades_365d", "win_rate")},
            }
            return metrics, rec

        if round_timeout_sec and round_timeout_sec > 0:
            import threading as _threading
            result_box: Dict[str, Any] = {}

            def _runner():
                try:
                    result_box["ok"] = _round_work()
                except BaseException as e:
                    result_box["err"] = e

            th = _threading.Thread(target=_runner, daemon=True)
            th.start()
            th.join(timeout=round_timeout_sec)
            if th.is_alive():
                # 執行緒是 daemon，泄漏一輪的等待可接受；上限先記錄再繼續。
                _cb(round_no, f"timeout_after_{int(round_timeout_sec)}s")
                history.append({"round": round_no, "spec_id": None, "fitness": None,
                                "accepted": False,
                                "error": f"round_timeout: exceeded {round_timeout_sec}s",
                                "request": request_text})
                continue
            if "err" in result_box:
                history.append({"round": round_no, "spec_id": None, "fitness": None,
                                "accepted": False,
                                "error": f"round_failed: {result_box['err']}",
                                "request": request_text})
                continue
            metrics, rec = result_box["ok"]
        else:
            metrics, rec = _round_work()

        _cb(round_no, f"pipeline_done: fitness={metrics.get('fitness')}")
        history.append(rec)
        try:
            if history_cb is not None:
                history_cb(deepcopy(history), deepcopy(best_metrics))
        except Exception:
            pass
        best_fitness = float(best_metrics.get("fitness", -1e9) or -1e9)
        rec["delta_fitness"] = round(rec["fitness"] - best_fitness, 6)
        rec["changed"] = {"request": request_text}
        if rec["accepted"]:
            best_metrics = metrics

    return {
        "best_spec": deepcopy(best_spec),
        "best_metrics": deepcopy(best_metrics),
        "rounds": history,
        "baseline_fitness": baseline_metrics.get("fitness"),
    }

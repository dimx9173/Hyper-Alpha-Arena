"""Training agent loop：LLM 依 baseline + 評估摘要提出下一個 StrategySpec。

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
每輪：build_prompt -> _call_llm -> parse_candidate ->（非法則帶錯誤回饋重試
同輪一次，仍非法則跳過）-> evaluate_spec_individual -> fitness 與最佳比較，
保留較佳者。本模組為新增檔案，不修改任何既有檔案。
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
Hard rules:
1. Output ONE valid JSON object only — no explanation, no markdown outside the JSON.
2. Every param within the ranges above.
3. Between 1 and 4 filters (max 4).
4. Change something meaningful vs. the baseline (structure or tuning)."""

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

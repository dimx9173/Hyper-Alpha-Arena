"""Trade-agent prompt compiler：LLM 寫規則文字 -> 結構化 filters -> 可回測策略碼。

Half 2 of training agent design：LLM 只寫交易規則的自然語言句子
（rule template），本模組把規則文字編譯成確定性的 sandbox-ready
策略碼。回測跑的是編譯後的 code，不是 LLM 原文。

輸入兩種形式：
  (a) JSON spec 字串：直接走 StrategySpec.from_dict（與 training_agent
      的 parse_candidate 同語義）；
  (b) 規則句子：四種已知 kind（ema_cross / donchian_breakout /
      rsi_band / ict_confirm），用 regex 抽參數，容忍散文噪音。

未知/無法解析的規則一律記入 meta["skipped"]，絕不拋例外；只有
generate_and_compile 在零可用 filter 時拋 RuntimeError（由 caller
決定 fallback）。

僅依賴 strategy_spec 的公開接縫（StrategySpec / FilterSpec /
RiskSpec / FILTER_PARAM_RANGES / render_strategy_from_spec）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from backtest.strategy_spec import (
        FILTER_PARAM_RANGES,
        FilterSpec,
        RiskSpec,
        StrategySpec,
        render_strategy_from_spec,
    )
except ImportError:  # pragma: no cover - 非標準路徑下的退路
    from backend.backtest.strategy_spec import (  # type: ignore
        FILTER_PARAM_RANGES,
        FilterSpec,
        RiskSpec,
        StrategySpec,
        render_strategy_from_spec,
    )


# ========== JSON 偵測（與 training_agent.parse_candidate 同語義） ==========

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    return m.group(1) if m else text


def _try_parse_json_spec(text: str) -> Optional[StrategySpec]:
    """若文字內含合法 spec JSON，回傳 StrategySpec；否則回傳 None。"""
    body = _strip_fences(text)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(body[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "filters" not in obj:
        return None
    try:
        return StrategySpec.from_dict(obj)
    except (KeyError, TypeError, AttributeError):
        return None


# ========== 規則句子解析 ==========

def _clamp(kind: str, name: str, value: float) -> Any:
    lo, hi = FILTER_PARAM_RANGES[kind][name]
    v = max(lo, min(hi, value))
    if name in ("ema_fast", "ema_slow", "lookback", "low", "high",
                "swing_len", "require_alignment", "require_bull_structure",
                "require_discount"):
        return int(round(v))
    return float(v)


def _parse_ema(line: str) -> Optional[Dict[str, Any]]:
    m = re.search(
        r"ema\D*?(\d{1,3})\D+?ema\D*?(\d{1,3})", line, re.IGNORECASE)
    if m:
        fast, slow = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(
            r"fast\D*?(\d{1,3}).*?slow\D*?(\d{1,3})", line, re.IGNORECASE)
        if not m2:
            return None
        fast, slow = int(m2.group(1)), int(m2.group(2))
    if fast > slow:  # 防呆：快的必須是小數字
        fast, slow = slow, fast
    return {"ema_fast": _clamp("ema_cross", "ema_fast", fast),
            "ema_slow": _clamp("ema_cross", "ema_slow", slow)}


def _parse_donchian(line: str) -> Optional[Dict[str, Any]]:
    lowered = line.lower()
    if not ("donchian" in lowered or "breakout" in lowered
            or ("bar" in lowered and ("high" in lowered or "close" in lowered))
            or "highest" in lowered):
        return None
    params: Dict[str, Any] = {}
    m = re.search(r"lookback\D*?(\d{1,3})", line, re.IGNORECASE)
    if not m:
        m = re.search(r"donchian\D*?(\d{1,3})", line, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{1,3})\s*[- ]?bar", line, re.IGNORECASE)
    if m:
        params["lookback"] = _clamp(
            "donchian_breakout", "lookback", int(m.group(1)))
    m = re.search(r"(0\.\d+)", line)
    if m:
        params["threshold"] = _clamp(
            "donchian_breakout", "threshold", float(m.group(1)))
    if not params:
        return None
    return params


def _parse_rsi(line: str) -> Optional[Dict[str, Any]]:
    if "rsi" not in line.lower():
        return None
    m = re.search(r"rsi\D*?(\d{1,3})\D+?(\d{1,3})", line, re.IGNORECASE)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(r"low\D*?(\d{1,3}).*?high\D*?(\d{1,3})",
                       line, re.IGNORECASE)
        if not m2:
            return None
        low, high = int(m2.group(1)), int(m2.group(2))
    if low > high:
        low, high = high, low
    return {"low": _clamp("rsi_band", "low", low),
            "high": _clamp("rsi_band", "high", high)}


def _parse_ict(line: str) -> Optional[Dict[str, Any]]:
    lowered = line.lower()
    if not (re.search(r"\bict\b", lowered) or re.search(r"\bote\b", lowered)):
        return None
    params: Dict[str, Any] = {}
    m = re.search(r"ote\D*?(0\.\d+|[01](?:\.\d+)?)", line, re.IGNORECASE)
    if m:
        try:
            params["ote_threshold"] = _clamp(
                "ict_confirm", "ote_threshold", float(m.group(1)))
        except ValueError:
            pass
    if "align" in lowered:
        params["require_alignment"] = 1
    if "bull" in lowered or "structure" in lowered:
        params["require_bull_structure"] = 1
    if "discount" in lowered or "premium" in lowered:
        params["require_discount"] = 1
    m = re.search(r"swing\D*?(\d{1,2})", line, re.IGNORECASE)
    if m:
        params["swing_len"] = _clamp(
            "ict_confirm", "swing_len", int(m.group(1)))
    m = re.search(r"weight\D*?(0\.\d+|1(?:\.0+)?|0(?:\.0+)?)",
                  line, re.IGNORECASE)
    if m:
        params["weight"] = _clamp("ict_confirm", "weight", float(m.group(1)))
    if not params:
        # 提到 ICT 但無具體參數：預設要求 alignment，讓 filter 有意義。
        params["require_alignment"] = 1
    return params


_LINE_PARSERS: Tuple[Tuple[str, Callable[[str], Optional[Dict[str, Any]]]], ...] = (
    ("ema_cross", _parse_ema),
    ("donchian_breakout", _parse_donchian),
    ("rsi_band", _parse_rsi),
    ("ict_confirm", _parse_ict),
)


def _parse_rule_lines(text: str) -> Tuple[List[FilterSpec], List[str]]:
    """逐行套用四種 parser；一行可命中多種；全無命中則記 skipped。"""
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    skipped: List[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("-*•> \t")
        if not line:
            continue
        hits = 0
        for kind, fn in _LINE_PARSERS:
            try:
                params = fn(line)
            except (ValueError, IndexError):
                params = None
            if params:
                hits += 1
                if kind not in merged:
                    merged[kind] = {}
                    order.append(kind)
                merged[kind].update(params)
        if hits == 0:
            skipped.append(raw.strip()[:160])
    return ([FilterSpec(kind=k, params=merged[k]) for k in order], skipped)


# ========== 風險參數正規化 ==========

def _normalize_risk(risk: Any) -> RiskSpec:
    if risk is None:
        return RiskSpec()
    if isinstance(risk, RiskSpec):
        return risk
    if isinstance(risk, dict):
        try:
            return RiskSpec.from_dict(risk)
        except TypeError:
            return RiskSpec()
    return RiskSpec()


# ========== 公開 API ==========

def compile_prompt_rules(
    rules_text: str,
    symbols: Any,
    kline_period: str = "1h",
    risk: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    """把 LLM 規則文字編譯成策略碼。

    JSON spec 字串走 StrategySpec.from_dict；規則句子走行解析。
    未知/無法解析的規則記入 meta["skipped"]，絕不拋例外。

    回傳 (code_str, meta)，meta 含 parsed_filters / skipped / spec_id。
    """
    spec: Optional[StrategySpec] = None
    skipped: List[str] = []
    parsed_filters: List[Dict[str, Any]] = []

    candidate = _try_parse_json_spec(rules_text)
    if candidate is not None:
        if risk is not None:
            candidate = StrategySpec(filters=candidate.filters,
                                     risk=_normalize_risk(risk),
                                     logic=candidate.logic)
        try:
            code = render_strategy_from_spec(
                candidate, symbols, kline_period)
        except (ValueError, KeyError, TypeError):
            candidate = None  # 非法 kind 等：退回句子解析
        else:
            spec = candidate
            parsed_filters = [f.to_dict() for f in spec.filters]

    if spec is None:
        filters, skipped = _parse_rule_lines(rules_text)
        spec = StrategySpec(filters=filters, risk=_normalize_risk(risk),
                            logic="all_pass")
        # JSON 偵測失敗但原文含 '{'：把整段也記一筆，方便除錯。
        parsed_filters = [f.to_dict() for f in spec.filters]
        code = render_strategy_from_spec(spec, symbols, kline_period)

    meta: Dict[str, Any] = {
        "parsed_filters": parsed_filters,
        "skipped": skipped,
        "spec_id": spec.spec_id(),
        "spec": spec.to_dict(),
    }
    return code, meta


RULES_SYSTEM_PROMPT = """You write ONLY trading rule sentences for a crypto backtest compiler. \
No JSON, no explanation, no markdown. One rule per line.

Allowed rule kinds (write only these four):
- EMA cross: e.g. "buy when ema12 crosses above ema55"
- Donchian breakout: e.g. "close > 55-bar high * 0.985"
- RSI band: e.g. "RSI between 30 and 70"
- ICT confirm: e.g. "require ICT alignment with OTE 0.3 and swing 5"

Ranges the compiler enforces (stay inside or values get clamped):
- ema_fast [10,30], ema_slow [40,80]
- donchian lookback [41,81], threshold [0.97,0.995]
- rsi low [25,45], high [55,75]
- ict ote_threshold [0.0,0.75], swing_len [1,6], weight [0.0,1.0]

Rendered-shape hint (what your rules compile to):
- class Strategy with should_trade(data) using data.get_klines / data.get_indicator / data.get_price
- all declared filters AND together (all_pass); each failing gate skips the symbol
- signal returns Decision(operation="buy", ...) with portion/leverage/take-profit/stop-loss

Write 1 to 4 rules. Prefer tuning the baseline structure over adding filters."""


def gen_trade_agent_rules(
    baseline_summary: Any,
    constraints_text: str,
    llm_call: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> str:
    """One-shot prompt：模型只寫規則句子（非 JSON），回傳原文。"""
    if llm_call is None:  # 延遲匯入，避免循環依賴
        try:
            from backtest.training_agent import _call_llm as _default
        except ImportError:
            from backend.backtest.training_agent import (  # type: ignore
                _call_llm as _default)
        llm_call = _default
    if isinstance(baseline_summary, dict):
        baseline_block = json.dumps(baseline_summary, sort_keys=True,
                                    ensure_ascii=False)
    else:
        baseline_block = str(baseline_summary)
    messages = [
        {"role": "system", "content": RULES_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Baseline summary:\n{baseline_block}\n\n"
            f"Constraints:\n{constraints_text}\n\n"
            "Write the improved rule set now (rule sentences only).")},
    ]
    return str(llm_call(messages))


def generate_and_compile(
    baseline_summary: Any,
    constraints_text: str,
    symbols: Any,
    kline_period: str = "1h",
) -> Tuple[str, Dict[str, Any], str]:
    """gen_trade_agent_rules 接著 compile_prompt_rules。

    零可用 filter 時拋 RuntimeError（內含 skipped 清單），由 caller
    決定 fallback（沿用 baseline / 重試 / 棄守）。
    """
    raw_text = gen_trade_agent_rules(baseline_summary, constraints_text)
    code, meta = compile_prompt_rules(raw_text, symbols, kline_period)
    if not meta["parsed_filters"]:
        raise RuntimeError(
            "no usable filters compiled; skipped=%r" % (meta["skipped"],))
    return code, meta, raw_text

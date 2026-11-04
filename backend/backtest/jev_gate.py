"""JEV pre-trade gate：通用轉換器 + QuantDinger 形狀的 SystemOne 呼叫。

設計說明
--------
- JEV（SystemOne）是核驗器，不是生成器：吃結構化 state + 五個 typed 問，
  回 typed choice + probabilities + confidence，不吃自由文字。
- trade agent 輸出的是 Decision（生成）。本模組把 Decision + MarketData
  翻成 JEV state（build_jev_state，通用：任何給出 symbol/operation/portion/
  leverage/price 的來源都能用），再交給 evaluate_gate 審。
- 呼叫形狀照抄 QuantDinger app/services/ai_decision_filter.py：
  POST {base_url}/systemone，body = {model, state, questions}，
  判定 = risk_check=='block' 或 execution_quality=='block' 或
  directional_block（signal conflict 且 regime adverse 且兩者 confidence
  >= min_confidence）→ reject。
- 無 JEV_API_KEY 時 fail open（回 True, "jev_not_configured"），與 QuantDinger
  一致；每次評估最多一次 HTTP 呼叫，timeout 可調。
- 兩種模式（JEV_GATE_MODE）：
  - "jes"（預設，真 SystemOne）：POST {base}/systemone，吃 typed 五問。
  - "llm"：base_url 指向普通 OpenAI 相容 gateway（例如 auto-free），
    把 state 文字翻成五個二選一 prompt 平行問模型，解析 verdict。
    適用於「模型不支援 SystemOne API」但仍能答題的情況。
- 本檔案是新增檔案，零修改既有檔案。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from program_trader.models import Decision
except ImportError:  # pragma: no cover - 非標準路徑下的退路
    from backend.program_trader.models import Decision  # type: ignore


# ========== JEV 問項（照抄 QuantDinger） ==========

JEV_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "data_quality": {
        "type": "choice",
        "instructions": (
            "Assess whether the supplied point-in-time evidence is sufficient "
            "for a pre-trade decision."
        ),
        "criteria": {
            "sufficient": "Market, signal, portfolio, and execution evidence "
                          "are current and usable.",
            "partial": "Some evidence is missing or stale, but concrete risk "
                       "checks remain possible.",
            "insufficient": "The state lacks enough current evidence for a "
                            "directional or risk judgement.",
        },
    },
    "signal_alignment": {
        "type": "choice",
        "instructions": (
            "Compare the requested action and signal reason with every "
            "available timeframe in context.market_evidence. Treat missing "
            "evidence as insufficient rather than conflict."
        ),
        "criteria": {
            "aligned": "Price trend, momentum, and volatility evidence "
                       "support the requested direction.",
            "mixed": "Evidence is usable but timeframes or indicators "
                     "disagree without a strong contradiction.",
            "conflict": "Current evidence materially contradicts the "
                        "requested direction or signal reason.",
            "insufficient": "There is not enough current market evidence to "
                            "judge alignment.",
        },
    },
    "market_regime": {
        "type": "choice",
        "instructions": (
            "Judge whether the current multi-timeframe market regime is "
            "suitable for this requested entry."
        ),
        "criteria": {
            "favorable": "Trend, momentum, volatility, and volume are "
                         "reasonably supportive of this entry.",
            "neutral": "The regime is mixed or range-bound but does not "
                       "materially oppose the entry.",
            "adverse": "The regime materially opposes the entry or shows "
                       "unstable conditions for it.",
            "insufficient": "The supplied market evidence is unavailable or "
                            "too stale to judge the regime.",
        },
    },
    "risk_check": {
        "type": "choice",
        "instructions": (
            "Assess position sizing, leverage, existing exposure, drawdown, "
            "recent realized performance, and margin headroom for this "
            "requested entry."
        ),
        "criteria": {
            "clear": "Sizing, leverage, exposure, and margin headroom are "
                     "within policy for this entry.",
            "caution": "One risk dimension is stretched but not breaching "
                       "policy; entry may proceed with reduced size.",
            "block": "Sizing, leverage, exposure, drawdown, or margin "
                     "breaches policy and the entry should not proceed.",
            "insufficient": "Portfolio or execution evidence is missing, so "
                            "risk cannot be judged.",
        },
    },
    "execution_quality": {
        "type": "choice",
        "instructions": (
            "Assess order quality: spread/SL/TP relative to reference price, "
            "price stability, and liquidity implied by notional vs volume."
        ),
        "criteria": {
            "clear": "Order quality is within normal bounds for this entry.",
            "caution": "One execution dimension is degraded but workable.",
            "block": "Execution would be unsafe (e.g. SL too far from "
                     "reference price for current stability, or notional "
                     "overwhelms liquidity).",
            "insufficient": "Execution evidence is missing.",
        },
    },
}

JEV_CHECK_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "data_quality": ("sufficient", "partial", "insufficient"),
    "signal_alignment": ("aligned", "mixed", "conflict", "insufficient"),
    "market_regime": ("favorable", "neutral", "adverse", "insufficient"),
    "risk_check": ("clear", "caution", "block", "insufficient"),
    "execution_quality": ("clear", "caution", "block", "insufficient"),
}

ENTRY_ACTIONS = {"open_long", "open_short", "add_long", "add_short", "buy", "sell"}


# ========== config ==========

def load_jev_config(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """JEV 設定：從環境變數讀，可傳自訂 env dict 測試。

    Key 放 repo 根目錄 `.env`（跟 DATABASE_URL / HYPERLIQUID_ENCRYPTION_KEY
    同一檔）；backend/main.py 的 load_dotenv() 會載入，shell 版用
    `set -a && . .env && set +a`。

    兩種模式（JEV_GATE_MODE）：
    - "jes"（預設，真 SystemOne）：POST {base}/systemone，吃 typed 五問。
    - "llm"：base_url 指向普通 OpenAI 相容 gateway（例如 auto-free），
      把 state 文字翻成五個二選一 prompt 平行問模型，解析 verdict。
      適用於「模型不支援 SystemOne API」但仍能答題的情況。
    JEV_API_KEY 缺席 → configured=False → evaluate_gate fail open。
    """
    source = env if env is not None else os.environ
    api_key = (source.get("JEV_API_KEY") or "").strip()
    return {
        "configured": bool(api_key),
        "api_key": api_key,
        "base_url": (source.get("JEV_BASE_URL") or
                     "https://api.typesafe.ai/v1").strip(),
        "model": (source.get("JEV_MODEL") or "jev-latest").strip(),
        "gate_mode": (source.get("JEV_GATE_MODE") or "jes").strip().lower(),
        "timeout_seconds": float(source.get("JEV_TIMEOUT_SECONDS") or 8),
        "min_confidence": float(source.get("JEV_MIN_CONFIDENCE") or 0.55),
    }


# ========== 通用轉換器 ==========

def build_jev_state(decision: Any, market_snapshot: Any) -> Dict[str, Any]:
    """把任意「決策 + 行情」翻成 QuantDinger 同形狀的 JEV state。

    通用性契約（duck-typed，不鎖特定類別）：
    - decision：任何有 operation/symbol（或 action）/portion/leverage/price
      屬性的物件，或同形狀 dict。屯 program_trader.Decision、自訂 dataclass、
      外部 trade agent 的輸出 dict 皆可。
    - market_snapshot：任何有 get_price/get_market_data/positions 的物件
      （program_trader.MarketData），或同形狀 dict：
      {"price": ..., "positions": {sym: {"side","size","entry_price"}},
       "equity": ..., "used_margin": ..., "market_evidence": {...}}。

    回傳：QuantDinger _state_payload 同形狀 dict：
    {source_type, symbol, action, market_type, order_type, quantity,
     reference_price, notional, leverage, strategy_type, signal_reason,
     context}
    """
    d = decision if isinstance(decision, dict) else getattr(decision, "to_dict", lambda: {})()
    if not d:
        d = {k: getattr(decision, k, None) for k in (
            "operation", "symbol", "target_portion_of_balance",
            "leverage", "max_price", "min_price", "reason", "trading_strategy")}
        d = {k: v for k, v in d.items() if v is not None}

    action_raw = str(d.get("operation") or d.get("action") or "hold").lower()
    symbol = str(d.get("symbol") or "")
    portion = float(d.get("portion") or d.get("target_portion_of_balance") or 0.0)
    leverage = max(1, int(d.get("leverage") or 1))

    # action 正規化：buy→open_long、sell→open_short、close 保留
    action = {
        "buy": "open_long",
        "sell": "open_short",
        "close": "close",
    }.get(action_raw, action_raw)

    # 價位：buy 用 max_price、sell/close 用 min_price，全缺則回 0（JEV 會判 insufficient）
    price = d.get("max_price") if action in ("open_long", "buy") else d.get("min_price")
    if price is None:
        price = d.get("price") or d.get("reference_price") or 0.0
    price = float(price)

    # quantity：可用顯示欄位或 portion ×在世界不在本函式內 — 用 equity 由 caller 算，
    # 這裡保留 notional 由 quantity × price 推得
    quantity = float(d.get("quantity") or 0.0)
    notional = float(d.get("notional") or quantity * price)

    ms = market_snapshot
    positions: Dict[str, Any] = {}
    equity = used_margin = 0.0
    market_evidence: Dict[str, Any] = {}
    order_type = "limit"
    if ms is not None:
        if isinstance(ms, dict):
            positions = ms.get("positions") if isinstance(ms.get("positions"), dict) else {}
            equity = float(ms.get("equity") or 0.0)
            used_margin = float(ms.get("used_margin") or 0.0)
            market_evidence = ms.get("market_evidence") if isinstance(ms.get("market_evidence"), dict) else {}
        else:
            for sym, pos in dict(getattr(ms, "positions", {}) or {}).items():
                positions[sym] = {
                    "side": getattr(pos, "side", None),
                    "size": getattr(pos, "size", None),
                    "entry_price": getattr(pos, "entry_price", None),
                }
            for getter, key in (
                ("get_market_data", "market"),
                ("get_regime", "regime"),
                ("get_indicator", "indicators"),
            ):
                try:
                    fn = getattr(ms, getter, None)
                    if fn is not None:
                        if getter == "get_market_data":
                            market_evidence[key] = fn(symbol) if symbol else {}
                        elif getter == "get_regime":
                            regime = fn(symbol, "1h")
                            # Serialize RegimeInfo to dict
                            market_evidence[key] = {
                                "regime": getattr(regime, "regime", None),
                                "conf": getattr(regime, "conf", None),
                                "direction": getattr(regime, "direction", None),
                                "reason": getattr(regime, "reason", None),
                                "indicators": getattr(regime, "indicators", None) or {},
                            }
                        else:
                            market_evidence[key] = fn(symbol, "1h")
                except Exception:
                    pass

            # Extract equity and used_margin from MarketData object attributes
            equity = float(getattr(ms, "total_equity", 0.0) or 0.0)
            used_margin = float(getattr(ms, "used_margin", 0.0) or 0.0)

    exposure = {
        "equity": equity,
        "used_margin": used_margin,
        "open_positions": len(positions),
        "positions": positions,
    }

    state: Dict[str, Any] = {
        "source_type": "program_strategy",
        "symbol": symbol,
        "action": action,
        "market_type": "perpetual_futures",
        "order_type": order_type,
        "quantity": quantity,
        "reference_price": price,
        "notional": notional,
        "leverage": leverage,
        "strategy_type": "programmatic",
        "signal_reason": str(d.get("reason") or d.get("signal_reason") or ""),
        "context": {
            "market_evidence": market_evidence,
            "exposure": exposure,
        },
    }
    return json.loads(json.dumps(state, ensure_ascii=False, default=str))


def state_text(state: Dict[str, Any]) -> str:
    """QuantDinger _state_text 同形狀：compact JSON。"""
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)


def state_text_llm(state: Dict[str, Any]) -> str:
    """LLM-friendly formatted state text for _evaluate_gate_llm.
    Converts compact JSON into a readable narrative so the LLM can
    reason about risk/execution questions properly.
    """
    lines = []
    lines.append(f"Source: {state.get('source_type', 'unknown')}")
    lines.append(f"Symbol: {state.get('symbol', '?')}")
    lines.append(f"Action: {state.get('action', '?')}")
    lines.append(f"Market: {state.get('market_type', '?')}")
    lines.append(f"Order Type: {state.get('order_type', '?')}")
    lines.append(f"Leverage: {state.get('leverage', '?')}x")
    lines.append(f"Strategy: {state.get('strategy_type', '?')}")
    lines.append(f"Signal Reason: {state.get('signal_reason', '?')}")

    # Order details
    qty = state.get('quantity', 0)
    notional = state.get('notional', 0)
    ref_price = state.get('reference_price', 0)
    if qty > 0 and ref_price > 0:
        lines.append(f"Order: qty={qty:.6f} @ {ref_price:.2f} (notional ${notional:,.2f})")

    # Context
    ctx = state.get('context', {})
    me = ctx.get('market_evidence', {})
    if me.get('market'):
        m = me['market']
        lines.append(f"Market: price=${m.get('price', '?')}, 24h chg={m.get('percentage24h', '?')}%, vol=${m.get('volume24h', '?'):,.0f}, OI=${m.get('open_interest', '?'):,.0f}, funding={m.get('funding_rate', '?')}")
    if me.get('regime'):
        r = me['regime']
        if isinstance(r, dict):
            lines.append(f"Regime: {r.get('regime', '?')} (conf={r.get('conf', '?'):.2f}, dir={r.get('direction', '?')}) - {r.get('reason', '')[:80]}")
        else:
            r_regime = getattr(r, 'regime', '?')
            r_conf = getattr(r, 'conf', None)
            r_dir = getattr(r, 'direction', '?')
            r_reason = getattr(r, 'reason', '')
            conf_str = f"{r_conf:.2f}" if isinstance(r_conf, (int, float)) else str(r_conf)
            lines.append(f"Regime: {r_regime} (conf={conf_str}, dir={r_dir}) - {str(r_reason)[:80]}")

    # Exposure - critical for risk_check
    exp = ctx.get('exposure', {})
    equity = exp.get('equity', 0)
    used = exp.get('used_margin', 0)
    open_pos = exp.get('open_positions', 0)
    margin_avail = equity - used
    margin_pct = (margin_avail / equity * 100) if equity > 0 else 0
    lines.append(f"Portfolio: equity=${equity:,.2f}, used_margin=${used:,.2f}, available=${margin_avail:,.2f} ({margin_pct:.1f}%), open_positions={open_pos}")
    if exp.get('positions'):
        for sym, pos in exp['positions'].items():
            lines.append(f"  {sym}: {pos.get('side', '?')} {pos.get('size', '?')} @ {pos.get('entry_price', '?')}")

    return "\n".join(lines)


def _answer(answers: Any, key: str) -> Dict[str, Any]:
    if isinstance(answers, dict):
        value = answers.get(key)
        if isinstance(value, dict):
            return value
        nested = answers.get("answers")
        if isinstance(nested, dict) and isinstance(nested.get(key), dict):
            return nested[key]
    return {}


# ========== gate ==========


def evaluate_gate(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[float], List[Dict[str, Any]]]:
    """JEV 審查。回 (allowed, reason, confidence, checks)。

    - 無 api_key → fail open：(True, "jev_not_configured", None, [])
    - risk_check=='block' 或 execution_quality=='block' 或
      directional_block → (False, "jev_entry_rejected:*")
    - 其餘 → (True, "jev_entry_approved")
    - HTTP/解析錯誤 → fail open + reason "jev_error_fail_open"
    """
    config = config or load_jev_config()
    if not config.get("configured") or not config.get("api_key"):
        return True, "jev_not_configured", None, []

    mode = config.get("gate_mode", "jes")
    if mode == "llm":
        return _evaluate_gate_llm(state, config)

    # JES (SystemOne) path
    url = config["base_url"].strip().rstrip("/")
    url = url if url.endswith("/systemone") else f"{url}/systemone"

    payload = {
        "model": config.get("model") or "jev-latest",
        "state": state_text(state),
        "questions": JEV_QUESTIONS,
    }
    headers = {
        "Authorization": f"Bearer {config.get('api_key', '')}",
        "Content-Type": "application/json",
    }

    import requests
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=max(1.0, min(float(config.get("timeout_seconds") or 8), 30.0)),
        )
        response.raise_for_status()
        body = response.json()
    except Exception as e:
        return True, f"jev_error_fail_open: {type(e).__name__}", None, []

    answers = body.get("answers") or body.get("result") or body.get("data") or {}
    results: Dict[str, Tuple[str, Dict[str, float], Optional[float]]] = {}
    checks: List[Dict[str, Any]] = []
    min_conf = max(0.0, min(float(config.get("min_confidence") or 0.55), 1.0))
    for name, options in JEV_CHECK_OPTIONS.items():
        answer = _answer(answers, name)
        choice, probabilities = _validate_choice(answer, name, options)
        confidence = _confidence(answer, probabilities, choice)
        results[name] = (choice, probabilities, confidence)
        checks.append({
            "name": name,
            "result": choice,
            "confidence": confidence,
            "probabilities": probabilities,
        })

    for name in ("risk_check", "execution_quality"):
        conf = results[name][2]
        if conf is None or conf < min_conf:
            return (False, f"jev_rejected:low_confidence_{name}", conf, checks)

    risk_choice, _, risk_conf = results["risk_check"]
    exec_choice, _, exec_conf = results["execution_quality"]
    signal_choice, _, signal_conf = results["signal_alignment"]
    regime_choice, _, regime_conf = results["market_regime"]

    directional_block = (
        signal_choice == "conflict"
        and regime_choice == "adverse"
        and float(signal_conf or 0) >= min_conf
        and float(regime_conf or 0) >= min_conf
    )
    allowed = (
        risk_choice != "block"
        and exec_choice != "block"
        and not directional_block
    )
    if allowed:
        reason = "jev_entry_approved"
    elif risk_choice == "block":
        reason = "jev_entry_rejected:risk_block"
    elif exec_choice == "block":
        reason = "jev_entry_rejected:execution_block"
    elif directional_block:
        reason = "jev_entry_rejected:signal_conflict"
    else:
        reason = "jev_entry_rejected"
    if risk_choice == "block":
        confidence = risk_conf
    elif exec_choice == "block":
        confidence = exec_conf
    elif directional_block:
        confidence = min(float(signal_conf or 0), float(regime_conf or 0))
    else:
        confidence = min(float(risk_conf or 0), float(exec_conf or 0))
    return allowed, reason, confidence, checks


def _evaluate_gate_llm(
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[bool, str, Optional[float], List[Dict[str, Any]]]:
    """LLM fallback：把 state 文字送給普通 OpenAI 相容 endpoint，
    五問平行問，解析 verdict。"""
    import requests
    text = state_text_llm(state)
    url = config["base_url"].strip().rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    model = config.get("model") or "auto-free"
    timeout = max(1.0, min(float(config.get("timeout_seconds") or 8), 30.0))

    results: Dict[str, Tuple[str, Dict[str, float], Optional[float]]] = {}
    checks: List[Dict[str, Any]] = []
    min_conf = max(0.0, min(float(config.get("min_confidence") or 0.55), 1.0))

    for name, q in JEV_QUESTIONS.items():
        opts = "/".join(q["criteria"].keys())
        prompt = f"""State: {text}

Question: {q['instructions']}
Options: {opts}
Reply with EXACTLY one of the options, no other text."""

        try:
            r = requests.post(
                url,
                headers=headers,
                json={"model": model, "messages": [
                    {"role": "system", "content": "You are a pre-trade risk gate. Output one option only."},
                    {"role": "user", "content": prompt}
                ], "max_tokens": 20, "temperature": 0.0},
                timeout=timeout,
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]["message"]["content"].strip().lower()
        except Exception:
            choice = "insufficient"

        if choice not in q["criteria"]:
            choice = "insufficient"
        probs = {choice: 1.0}
        conf = 1.0 if choice != "insufficient" else 0.0
        results[name] = (choice, probs, conf)
        checks.append({
            "name": name,
            "result": choice,
            "confidence": conf,
            "probabilities": probs,
        })

    # reuse JES decision logic
    for nm in ("risk_check", "execution_quality"):
        conf = results[nm][2]
        if conf is None or conf < min_conf:
            return (False, f"jev_rejected:low_confidence_{nm}", conf, checks)

    risk_choice, _, risk_conf = results["risk_check"]
    exec_choice, _, exec_conf = results["execution_quality"]
    signal_choice, _, signal_conf = results["signal_alignment"]
    regime_choice, _, regime_conf = results["market_regime"]

    directional_block = (
        signal_choice == "conflict"
        and regime_choice == "adverse"
        and float(signal_conf or 0) >= min_conf
        and float(regime_conf or 0) >= min_conf
    )
    allowed = (
        risk_choice != "block"
        and exec_choice != "block"
        and not directional_block
    )
    if allowed:
        reason = "jev_entry_approved"
    elif risk_choice == "block":
        reason = "jev_entry_rejected:risk_block"
    elif exec_choice == "block":
        reason = "jev_entry_rejected:execution_block"
    elif directional_block:
        reason = "jev_entry_rejected:signal_conflict"
    else:
        reason = "jev_entry_rejected"
    if risk_choice == "block":
        confidence = risk_conf
    elif exec_choice == "block":
        confidence = exec_conf
    elif directional_block:
        confidence = min(float(signal_conf or 0), float(regime_conf or 0))
    else:
        confidence = min(float(risk_conf or 0), float(exec_conf or 0))
    return allowed, reason, confidence, checks


def _answer(answers: Any, key: str) -> Dict[str, Any]:
    if isinstance(answers, dict):
        value = answers.get(key)
        if isinstance(value, dict):
            return value
        nested = answers.get("answers")
        if isinstance(nested, dict) and isinstance(nested.get(key), dict):
            return nested[key]
    return {}


def _probabilities(answer: Dict[str, Any]) -> Dict[str, Any]:
    value = answer.get("probabilities") or answer.get("probability") or {}
    return dict(value) if isinstance(value, dict) else {}


def _validate_choice(
    answer: Dict[str, Any],
    question: str,
    options: Tuple[str, ...],
) -> Tuple[Optional[str], Dict[str, float]]:
    """QuantDinger _validate_choice_answer 同語義：先 choice 再機率 argmax。
    全缺 → (None, {})；驗證後 caller 走 low-confidence/fail path。
    """
    choice = answer.get("choice") or answer.get("result") or answer.get("answer")
    if isinstance(choice, str) and choice.strip() in options:
        return choice.strip(), _probabilities(answer)
    probs = _probabilities(answer)
    if probs:
        valid = {k: float(v) for k, v in probs.items() if k in options and isinstance(v, (int, float))}
        if valid:
            best = max(valid, key=valid.get)
            return best, valid
    return None, {}


def _confidence(
    answer: Dict[str, Any],
    probabilities: Dict[str, float],
    choice: Optional[str],
) -> Optional[float]:
    """QuantDinger 同語義：機率總和/最大值/choice 直接給值，擇一。"""
    explicit = answer.get("confidence")
    if isinstance(explicit, (int, float)):
        return float(explicit)
    if probabilities and choice:
        vals = [float(v) for v in probabilities.values()]
        total = sum(vals)
        if total > 0 and choice in probabilities:
            return float(probabilities[choice]) / total
    return None
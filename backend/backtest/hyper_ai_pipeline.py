"""
Offline Hyper AI Pipeline
=========================
Wrap the 4 Hyper AI sub-agents (Prompt AI → Program AI → Signal AI → Attribution AI)
into a single callable function that runs without SSE streaming, returns the final
strategy code that can be evaluated by evaluate_spec_individual / ProgramBacktestEngine.

Usage:
    from backtest.hyper_ai_pipeline import run_hyper_ai_offline
    code, trace = run_hyper_ai_offline(
        user_request="EMA cross + Donchian breakout + RSI band on BTC 1h",
        exchange="hyperliquid",
        symbols=["BTC"]
    )
    # code -> strategy python code string
    # trace -> dict with intermediate outputs for debugging/analysis
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

# Ensure backend is on path
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from database.connection import SessionLocal
from database.models import Account, TradingProgram, SignalPool
from services.hyper_ai_service import get_llm_config


@contextmanager
def _db_session():
    """Context manager for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_direct_completion(task: str, llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """單次 chat completion（無工具）：離線訓練最穩定的代碼來源。

    Program AI 的工具迴圈（15 輪 x 每輪 LLM+工具 60-180s）對網關抖動極敏感，
    v10-v12 有 4/5、5/5、3/5 的輪次死在讀超時。直接要 fenced code：
    一次請求、一次回覆（通常 15-60s），回來就抽 class Strategy。
    requests 拋錯 → 呼叫端記 round_failed，不偽裝成静默垃圾。
    """
    import requests as _requests
    from services.ai_decision_service import (
        build_chat_completion_endpoints, build_llm_payload,
        build_llm_headers, detect_api_format,
    )
    base_url = llm_config.get("base_url", "")
    api_format = llm_config.get("api_format") or "openai"
    endpoint, detected = detect_api_format(base_url)
    if detected:
        api_format = detected
    if api_format == "anthropic":
        endpoint = endpoint or (base_url.rstrip("/") + "/v1/messages")
        endpoints = [endpoint]
    else:
        endpoints = build_chat_completion_endpoints(base_url, llm_config.get("model"))
        if not endpoints:
            endpoints = [base_url.rstrip("/") + "/chat/completions"]
    payload = build_llm_payload(
        model=llm_config.get("model"),
        messages=[{"role": "user", "content": task}],
        api_format=api_format,
        max_tokens=4000,
        temperature=0.7,
        stream=True,
    )
    headers = build_llm_headers(api_format, llm_config.get("api_key"), base_url)
    # 非串流時網關要 85-97s 才吐第一個 byte，90s 讀超時幾乎必死（實測 3/3 timeout）。
    # 串流讓每個 SSE chunk 重置讀時鐘：只要上游持續產出就不會撞 timeout。
    # 仍保留 3 次嘗試：TCP 斷線、上游 5xx 都可重試。
    last_err = "no endpoint tried"
    attempts = 3
    for attempt in range(1, attempts + 1):
        for ep in endpoints:
            try:
                text_parts: List[str] = []
                calls: List[Dict[str, Any]] = []
                # OpenAI streaming 的 tool_calls 是分片送、以 index 累積 arguments。
                pending: Dict[int, Dict[str, Any]] = {}
                with _requests.post(ep, json=payload, headers=headers,
                                    timeout=(15, 120), stream=True) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        line = raw.strip()
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if not line or line == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            continue
                        if api_format == "anthropic":
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta") or {}
                                if delta.get("text"):
                                    text_parts.append(delta["text"])
                            continue
                        for choice in (chunk.get("choices") or []):
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                text_parts.append(str(delta["content"]))
                            for tc in (delta.get("tool_calls") or []):
                                idx = tc.get("index", 0)
                                slot = pending.setdefault(
                                    idx, {"tool": "", "args_raw": ""})
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["tool"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["args_raw"] += str(fn["arguments"])
                for _idx in sorted(pending):
                    slot = pending[_idx]
                    try:
                        args = json.loads(slot["args_raw"]) if slot["args_raw"] else {}
                    except Exception:
                        args = {"_raw": slot["args_raw"]}
                    calls.append({"tool": slot["tool"], "args": args})
                text = "".join(text_parts)
                if calls:
                    return {"content": text, "tool_calls_log": calls, "direct": True,
                            "endpoint": ep, "attempts": attempt}
                if text.strip():
                    return {"content": text, "tool_calls_log": [], "direct": True,
                            "endpoint": ep, "attempts": attempt}
                last_err = "empty stream"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"Direct completion failed after {attempts} attempts: {last_err}")


def _run_stream_to_completion(generator: Generator[str, None, None]) -> Dict[str, Any]:
    """Run a streaming generator to completion, return the final 'done' event data.

    done 本身不含 code_suggestion（Program AI 把 code 經 suggest_save_code
    工具存進 assistant_msg，done 只剩摘要 prose；見 v4 r1/r2 code_head）。
    所以順手收集最後一次 save_suggestion 的 code，存進 done["_saved_code"]。
    """
    last_done = None
    last_saved_code: Optional[str] = None
    for chunk in generator:
        try:
            # chunk is SSE format: "event: X\ndata: {...}\n\n"
            if chunk.startswith("event: save_suggestion"):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        data = payload.get("data") or {}
                        code = data.get("code") or ""
                        if code.strip():
                            last_saved_code = code
                        break
            elif chunk.startswith("event: done"):
                # Extract JSON after "data: "
                lines = chunk.strip().split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        last_done = json.loads(data_str)
                        break
        except (json.JSONDecodeError, IndexError):
            continue
    if last_done is None:
        raise RuntimeError("Stream did not produce a 'done' event")
    if last_saved_code:
        last_done["_saved_code"] = last_saved_code
    return last_done


def run_prompt_ai_offline(
    db,
    user_request: str,
    exchange: str = "hyperliquid",
    conversation_id: Optional[int] = None,
    prompt_id: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run Prompt AI offline, return (prompt_text, full_trace)."""
    from services.ai_prompt_generation_service import generate_prompt_with_ai_stream
    from services.hyper_ai_service import get_llm_config

    llm_config = get_llm_config(db)
    if not llm_config.get("configured"):
        raise RuntimeError("LLM not configured")

    gen = generate_prompt_with_ai_stream(
        db=db,
        user_message=user_request,
        conversation_id=conversation_id,
        prompt_id=prompt_id,
        llm_config=llm_config,
    )
    done = _run_stream_to_completion(gen)
    prompt_text = done.get("prompt_result") or ""
    if not prompt_text.strip():
        # Prompt AI 有時只回散文不包 ```prompt fence，導致 prompt_result=None。
        # 此時用 LLM 的完整回覆當作 spec：它包含策略描述，Program AI 能直接用。
        prompt_text = (done.get("content") or "").strip()
    return prompt_text, done


def run_program_ai_offline(
    db,
    prompt_text: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    conversation_id: Optional[int] = None,
    program_id: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run Program AI offline, return (strategy_code, full_trace)."""
    from services.ai_program_service import generate_program_with_ai_stream
    from services.hyper_ai_service import get_llm_config

    llm_config = get_llm_config(db)
    if not llm_config.get("configured"):
        raise RuntimeError("LLM not configured")

    # Sandbox 相容格式：只給「骨架 + API 清單」，不給可抄的完整範例邏輯。
    # 先前版本附了完整 EMA12/55+Donchian+RSI 範例，Program AI 直接原樣照抄，
    # 導致 5 輪訓練全部產出同一份代碼（fitness 幾乎不變）。
    task_template = f"""Write a NEW Python trading strategy for this specification:

--- SPECIFICATION START ---
{prompt_text}
--- SPECIFICATION END ---

Exchange: {exchange}
Symbols: {symbols or ['BTC']}
Kline period: 1h

The specification above is the ONLY source of trading logic. Implement exactly
what it says. Do NOT copy any example conditions.

## REQUIRED CLASS SHAPE

```python
class Strategy:
    def should_trade(self, data):
        # your logic
```

Class name MUST be exactly `Strategy`. Method MUST be exactly `should_trade(self, data)`.

## AVAILABLE API (this is the complete list — do not use anything else)

data.trigger_symbol                    -> str, symbol that fired the trigger
data.get_price(symbol)                 -> float, current price (0 if unavailable)
data.positions                         -> dict[str, Position]; Position has .size and .side ("long"/"short") and .leverage
data.get_indicator(symbol, name, tf)   -> dict

Supported `name` values for get_indicator:
  "EMA<N>"      any integer N, e.g. "EMA9", "EMA12", "EMA21", "EMA55", "EMA200"
                -> {{"value": float}}
  "MA<N>"       any integer N, e.g. "MA20", "MA50"
                -> {{"value": float}}
  "RSI14", "RSI7"                  -> {{"value": float}}
  "MACD"                           -> {{"macd": float, "signal": float, "histogram": float}}
  "BOLL"                           -> {{"upper": float, "middle": float, "lower": float}}
  "ATR14"                          -> {{"value": float}}
  "DONCHIAN<N>" any integer N      -> {{"upper": float, "middle": float, "lower": float}}
  "ICT"                            -> ICT payload dict (NO direction/bias key — do not gate on it)
  "ICT_OTE"                        -> {{"score": float, "direction": str, ...}}  <- use this for direction
  "STOCH"                          -> {{"k": float, "d": float}}
  "VWAP" / "OBV"                   -> {{"value": float}}

`tf` is the kline period, always pass "1h" here.

## DECISION OBJECT (return exactly one per call)

Decision(operation="hold", symbol=symbol, reason="...")
Decision(operation="buy",  symbol=symbol, target_portion_of_balance=<0.01-1.0>,
         leverage=<1-50>, max_price=<float>, reason="...", trading_strategy="...")
Decision(operation="sell", symbol=symbol, target_portion_of_balance=<0.01-1.0>,
         leverage=<1-50>, min_price=<float>, reason="...", trading_strategy="...")
Decision(operation="close", symbol=symbol, target_portion_of_balance=<0.01-1.0>,
         leverage=<existing>, min_price=<float for closing long>,
         reason="...")

## SANDBOX CONSTRAINTS (violating ANY of these makes the code invalid)

The sandbox runs your class with a restricted builtin set. These are HARD
failures, not style warnings:

1. `getattr`, `setattr`, `hasattr`, `eval`, `exec`, `compile`, `open`,
   `globals`, `locals`, `vars`, `input` are FORBIDDEN. Calling any of them
   aborts the whole backtest with "Validation failed: Forbidden function".
   Read attributes directly instead:
     WRONG:  getattr(pos, "size", 0) > 0
     RIGHT:  pos is not None and pos.size > 0
     WRONG:  getattr(pos, "leverage", 2)
     RIGHT:  pos.leverage if pos is not None else 2
2. `leverage` MUST be an INTEGER in Decision (1-50). `leverage=1.5` fails
   validation. Use 1, 2 or 3 — never a float, never a variable holding a float.
3. `target_portion_of_balance` MUST be a float in (0.0, 1.0].
4. Allowed builtins only: abs, min, max, sum, len, round, int, float, str,
   bool, list, dict, tuple, set, range, enumerate, zip, sorted, reversed,
   any, all, isinstance, type, print, True, False, None. `math` and `time`
   are pre-injected, no import needed.
5. Descriptive comments in Chinese or English are fine, but no prose paragraphs.

## HARD RULES

6. Never read an indicator you did not need; every get_indicator call costs time.
7. Guard missing indicator values: `.get("value", 0)` or `or 0`, never assume a key exists.
8. If you hold a position on the symbol, you must have an exit path (close on the
   inverse condition). Never leave a position with no exit rule.
9. Do NOT use data.get_market_data, data.get_klines, data.get_flow,
   data.get_regime, data.get_factor, or any API not listed above.
10. Do NOT return the sample trading logic from this message — it contains none.
    Your conditions must come from the SPECIFICATION block only.
11. If ICT confirmation is requested, use `ICT_OTE` and gate on
    `ict_data.get("direction")` + `ict_data.get("score")`. Never use raw `ICT`
    for direction — its direction key is always None, which turns the ICT gate
    permanently False and produces 0 trades.
12. `ICT_OTE.score` is ALWAYS in [0.0, 1.0] — it is confluence STRENGTH, never
    signed. Direction lives in `ict_data.get("direction")`: "bullish" or
    "bearish" (None when unconfirmed). A short entry must gate
    `direction == "bearish" and score >= threshold`, NEVER `score < -0.6`
    (score is never negative, so that branch can never fire → 0 trades).
13. Entries must test STATE, not an EVENT. Never gate an entry on
    `should_trade` detecting a cross that happened between calls (e.g.
    `crossed = prev != current`, storing `self.prev_*`). `should_trade` is
    invoked once per scheduled trigger, NOT once per bar, so a cross-event
    gate fires roughly once per position lifetime and yields ~1 trade per
    180 days. Compare current indicator ORDER instead:
      WRONG: crossed = (self.prev_above != above); if crossed and above: buy
      RIGHT: if ema_fast > ema_slow * tol and price >= donchian_upper: buy
14. Target 60-150 entries per year. Before returning, sanity-check that your
    gates can co-occur: an "all_pass" AND of 4 filters over an event-style
    condition typically yields 0-2 trades. Prefer state comparisons and
    tolerances (e.g. price >= upper * 0.99) over exact breakouts.
15. The Strategy object is REBUILT on every trigger — `self.*` does not persist.
    NEVER store history in instance attributes (self.prev_*, self.last_obv,
    self.obv_history, self.prices, ...). Anything written in one call is GONE
    by the next call, so such branches silently run on half-initialized
    constants every time.
      WRONG: self.obv_history.append(obv); slope = obv - self.obv_history[0]
      RIGHT: use same-trigger direction proxies (MACD histogram sign, VWAP
             side, BOLL position) and OBV only as an optional same-call
             sanity check (e.g. require obv value > 0 for longs).
16. OBV is CUMULATIVE volume — it has no fixed levels. Never compare it to an
    absolute threshold (e.g. `obv > 1000` is meaningless). If you must use it,
    only use its sign (obv value > 0 vs < 0) as a weak same-trigger filter.
17. MULTI-SYMBOL POLLING (required when Symbols lists more than one). The engine
    fires ONE scheduled trigger per interval with `data.trigger_symbol` set to
    Symbols[0] only. Indicators and prices ARE available for every symbol:
    `data.get_indicator("ETH", "STOCH", "1h")` works. Therefore a single-symbol
    body silently trades only Symbols[0] and wastes the rest of the universe.
    Correct shape — poll every symbol each call, exits first, then entries:
      SYMBOLS = ("BTC", "ETH", "SOL")
      for s in SYMBOLS:                      # 1) exit pass
          pos = data.positions.get(s)
          ...
      for s in SYMBOLS:                      # 2) entry pass
          if s in data.positions and data.positions[s].size > 0: continue
          ...
          return Decision(operation="buy", symbol=s, ...)
    Verified: same STOCH/BOLL logic went 34 trades (-32.8%) on one symbol to
    100 trades (-23.3%, PF 0.32 -> 0.56) when polled across three.

## HOW TO DELIVER THE CODE (this is the answer format)
"""
    # 離線評估外包給 harness：工具循環每輪都要 60-180s，網關讀超時（r2-r5）
    # 就直接殺掉整輪。direct=True（預設）走單次 chat completion，不用任何工具：
    # LLM 直接輸出 ```python fenced code，一次請求、一次回覆，不會超時。
    use_direct = True
    if use_direct:
        delivery_instruction = (
            "Output ONLY ONE ```python fenced code block containing the complete "
            "`class Strategy` implementation. No prose before or after the fence — "
            "no 'Looking at this specification', no plan, no summary. The fence is "
            "the whole answer."
        )
    else:
        delivery_instruction = (
            "Put the code body directly in the suggest_save_code `code` argument "
            "with NO markdown fences. Call suggest_save_code EXACTLY ONCE with "
            "the complete final code."
        )
    task = task_template + "\n" + delivery_instruction
    if use_direct:
        done = _run_direct_completion(task, llm_config)
    else:
        gen = generate_program_with_ai_stream(
            db=db,
            user_message=task,
            conversation_id=conversation_id,
            program_id=program_id,
            llm_config=llm_config,
            # 離線評估外包給 harness：工具只留 suggest_save_code（一寫即停），
            # 3 輪足夠（r5 成功只用 1 工具輪 + 存檔輪）。
            offline_mode=True,
            max_rounds=3,
        )
        done = _run_stream_to_completion(gen)

    # 1. Program AI 常把 code 當作工具參數（test_run_code / validate_code /
    #    quick_verify_strategy）送出，正文只剩摘要散文。v5 證明：save_suggestion
    #    有時根本沒被調用，但 tool_calls_log 裡的 args.code 就是同一份真代碼。
    code = _extract_code_from_tool_calls(done.get("tool_calls_log") or [])
    if code:
        return code, done

    # 2. 真正存下的建議（live 路徑常用）。
    saved = (done.get("_saved_code") or "").strip()
    if saved:
        return saved, done

    # 3. 回退：從 content 抽 fence（Program AI 直接把 code 寫在正文時）。
    content = done.get("content", "") or ""
    code = _extract_strategy_code(content)
    return code, done


def _strip_fences(code: str) -> str:
    """剝掉 Program AI 有時會加在 code 參數頭尾的 markdown 圍欄。

    r3 實證：args.code 以 ```python 開頭、``` 結尾，導致 validator 誤判前綴
    進而整個回測 0 筆交易。剝掉後才交給沙箱。
    """
    s = (code or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else ""
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()

_GETATTR_RE = re.compile(
    r'getattr\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*,\s*([^()]+?)\s*\)'
)


def _sanitize_code(code: str) -> str:
    """把 Program AI 常見的沙箱不合法寫法自動改寫成合法形式。

    1. getattr(obj, "attr", default) -> (obj.attr if obj is not None else default)
       getattr 在 FORBIDDEN_FUNCTIONS，但 LLM 防禦性程式碼必寫；機械改寫保住退款。
    2. leverage=<float> 常數 -> int 截斷（Decision 校驗要求 1-50 整數）。
    只處理字面量屬性名與單行 default 的形式；更複雜的呼叫讓沙箱報錯。
    """
    def _repl(m: "re.Match[str]") -> str:
        obj, attr, default = m.group(1), m.group(2), m.group(3).strip()
        return f"({obj}.{attr} if {obj} is not None else {default})"

    s = _GETATTR_RE.sub(_repl, code)
    s = re.sub(r"(leverage\s*=\s*)(\d+\.\d+)",
               lambda m: f"{m.group(1)}{max(1, min(50, int(float(m.group(2)))))}", s)
    return s


def _extract_code_from_tool_calls(tool_calls_log: Any) -> str:
    """從 tool_calls_log 的 args.code 取最長的 Strategy 實作。

    優先順序：suggest_save_code > quick_verify_strategy > test_run_code >
    validate_code；同名取參數最長者（Program AI 先測草稿、後存定稿）。
    非 Strategy 代碼（散文、空字串）直接丟棄。
    回傳前剝掉 markdown 圍欄。
    """
    if isinstance(tool_calls_log, str):
        try:
            tool_calls_log = json.loads(tool_calls_log)
        except Exception:
            return ""
    if not isinstance(tool_calls_log, list):
        return ""
    prio = {"suggest_save_code": 0, "quick_verify_strategy": 1,
            "test_run_code": 2, "validate_code": 3}
    cands: List[tuple] = []
    for entry in tool_calls_log:
        if not isinstance(entry, dict):
            continue
        name = entry.get("tool", "")
        if name not in prio:
            continue
        args = entry.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                continue
        if not isinstance(args, dict):
            continue
        code = args.get("code") or ""
        if "class Strategy" in code and "def should_trade" in code:
            cands.append((prio[name], -len(code), code))
    if not cands:
        return ""
    cands.sort()
    return _sanitize_code(_strip_fences(cands[0][2]))


_PY_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_strategy_code(content: str) -> str:
    """從 LLM 內容抽出 sandbox 相容的 Strategy 類別。

    優先順序：
    1. 含 "class Strategy" 且**有實質邏輯**（get_indicator 或 get_price 呼叫）的
       最後一個 fence —— Program AI 常先把我們給的空白骨架回貼一次，再給實作。
    2. 任一含 "class Strategy" 的 fence（取最後一個）
    3. 最後一個含 "def should_trade" 的 fence
    4. 最後一個 python fence
    5. 空字串 —— v13 教訓：direct completion 有時只回散文（"Looking at this
       specification..."）。把散文當 code 進沙箱會產生 fake fitness（0 筆
       交易靜默通過），寧可記 round_failed 讓訓練迴圈跳過。
    """
    blocks = [m.group(1) for m in _PY_FENCE_RE.finditer(content)]
    if not blocks:
        # 無 fence：內容若不含 Strategy 類別關鍵字 → 散文，拒收。
        if "class Strategy" not in content:
            return ""
        return content
    # 1. 有實質邏輯的實作（骨架 block 只有註解與 return hold）
    for b in reversed(blocks):
        if "class Strategy" in b and "def should_trade" in b and (
            "get_indicator(" in b or "get_price(" in b
        ):
            return b
    # 2. 任何 class Strategy block
    for b in reversed(blocks):
        if "class Strategy" in b and "def should_trade" in b:
            return b
    # 3. 任何 should_trade block
    for b in reversed(blocks):
        if "def should_trade" in b:
            return b
    # 4. 沒有任何實質程式碼的 fence —— v14 r-probe：模型回了一串
    #    ```...``` 包住的規格要點（"RSI band gates: - long: RSI14 between..."），
    #    blocks[-1] 把它當 code 送進沙箱，回測 0 筆交易且 fitness 假通過。
    return ""


def run_signal_ai_offline(
    db,
    user_request: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    conversation_id: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Signal AI offline, return (signal_config, full_trace)."""
    from services.ai_signal_generation_service import generate_signal_with_ai_stream
    from services.hyper_ai_service import get_llm_config

    llm_config = get_llm_config(db)
    if not llm_config.get("configured"):
        raise RuntimeError("LLM not configured")

    task = f"""Create signal pool for: {user_request}
Exchange: {exchange}
Symbols: {symbols or ['BTC']}
"""
    gen = generate_signal_with_ai_stream(
        db=db,
        user_message=task,
        conversation_id=conversation_id,
        llm_config=llm_config,
    )
    done = _run_stream_to_completion(gen)

    # Extract signal config
    signal_config = done.get("signal_config", {})
    return signal_config, done


def run_hyper_ai_offline(
    user_request: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    db_factory: Optional[Any] = None,
    use_prompt_ai: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the Hyper AI pipeline offline (Program AI directly by default).

    use_prompt_ai=False（預設）：Training Agent 的 LLM 請求就是乾淨的自然語言
    策略規格（entry/exit、指標、參數），直接餵給 Program AI。
    use_prompt_ai=True：走 Prompt AI → Program AI 兩段（產品 live 路徑用；
    Prompt AI 輸出的是帶 {runtime_minutes}/{trading_environment} 佔位符和
    live-money 警告的 session-templated prompt，Program AI 離線會誤讀為
    「規格為空」，只能產出 hold-stub — 見 /tmp/hyper_train_v3.json 五輪全 0 筆）。

    Returns (strategy_code, trace_dict).
    """
    db = (db_factory or SessionLocal)()
    trace = {
        "user_request": user_request,
        "exchange": exchange,
        "symbols": symbols or ["BTC"],
        "stages": {},
    }

    try:
        if use_prompt_ai:
            # Stage 1: Prompt AI（僅產品路徑用，offline 預設跳過）
            prompt_text, prompt_trace = run_prompt_ai_offline(db, user_request, exchange)
            trace["stages"]["prompt_ai"] = {
                "prompt_text": prompt_text,
                "trace": prompt_trace,
            }
        else:
            # Training Agent 的 request 本身就是規格，直接當作 prompt_text。
            prompt_text = user_request
            trace["stages"]["prompt_ai"] = {"skipped": True}

        # Stage 2: Program AI
        strategy_code, program_trace = run_program_ai_offline(
            db, prompt_text, exchange, symbols
        )
        trace["stages"]["program_ai"] = {
            "strategy_code": strategy_code,
            "trace": program_trace,
        }

        trace["status"] = "success"
        return strategy_code, trace

    except Exception as e:
        trace["status"] = "failed"
        trace["error"] = str(e)
        raise
    finally:
        db.close()


def strategy_code_to_spec(strategy_code: str) -> Dict[str, Any]:
    """
    Convert generated strategy code to a StrategySpec dict.
    This is a best-effort parser for the specific patterns our Program AI produces.
    Falls back to a generic spec if parsing fails.
    """
    import re
    from backtest.strategy_spec import StrategySpec, FilterSpec, RiskSpec, LogicSpec

    # Try to extract EMA/RSI/Donchian/ICT parameters from code
    spec_dict = {
        "filters": [],
        "risk": {
            "tp_pct": 0.08,
            "sl_pct": 0.04,
            "portion": 0.35,
            "leverage": 2,
            "max_positions": 3,
        },
        "logic": "all_pass",
    }

    # Extract ema_fast / ema_slow
    ema_fast_match = re.search(r"EMA\s*(\d+)", strategy_code, re.IGNORECASE)
    ema_slow_match = re.search(r"EMA\s*(\d+)", strategy_code, re.IGNORECASE)
    # More precise: look for ema_fast/ema_slow in params or comparisons
    ema_fast = re.search(r"ema_fast\s*[:=]\s*(\d+)", strategy_code)
    ema_slow = re.search(r"ema_slow\s*[:=]\s*(\d+)", strategy_code)
    if ema_fast and ema_slow:
        spec_dict["filters"].append({
            "kind": "ema_cross",
            "params": {
                "ema_fast": int(ema_fast.group(1)),
                "ema_slow": int(ema_slow.group(1)),
                "ema_tol_buy": 0.9985,
                "ema_tol_sell": 1.0008,
            },
        })

    # Extract Donchian
    dc_lookback = re.search(r"donchian.*?lookback\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    dc_thr = re.search(r"donchian.*?threshold\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    if dc_lookback and dc_thr:
        spec_dict["filters"].append({
            "kind": "donchian_breakout",
            "params": {
                "lookback": int(dc_lookback.group(1)),
                "threshold": float(dc_thr.group(1)),
            },
        })

    # Extract RSI
    rsi_low = re.search(r"rsi_low\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    rsi_high = re.search(r"rsi_high\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    if rsi_low and rsi_high:
        spec_dict["filters"].append({
            "kind": "rsi_band",
            "params": {
                "low": int(rsi_low.group(1)),
                "high": int(rsi_high.group(1)),
            },
        })

    # Extract ICT
    ict_w = re.search(r"ict_weight\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    if ict_w:
        spec_dict["filters"].append({
            "kind": "ict_confirm",
            "params": {
                "ote_threshold": float(ict_w.group(1)),
                "require_alignment": 0,
                "require_bull_structure": 0,
                "require_discount": 0,
                "swing_len": 5,
                "weight": float(ict_w.group(1)),
            },
        })

    # Extract risk params
    portion = re.search(r"portion\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    leverage = re.search(r"leverage\s*[:=]\s*(\d+)", strategy_code, re.IGNORECASE)
    tp = re.search(r"tp_pct\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    sl = re.search(r"sl_pct\s*[:=]\s*([\d.]+)", strategy_code, re.IGNORECASE)
    if portion:
        spec_dict["risk"]["portion"] = float(portion.group(1))
    if leverage:
        spec_dict["risk"]["leverage"] = int(leverage.group(1))
    if tp:
        spec_dict["risk"]["tp_pct"] = float(tp.group(1))
    if sl:
        spec_dict["risk"]["sl_pct"] = float(sl.group(1))

    # Validate via StrategySpec.from_dict
    try:
        spec = StrategySpec.from_dict(spec_dict)
        return spec.to_dict()
    except Exception:
        return spec_dict


def evaluate_hyper_ai_output(
    user_request: str,
    exchange: str = "hyperliquid",
    symbols: Optional[List[str]] = None,
    backtest_days: int = 180,
    end_time_ms: Optional[int] = None,
    kline_period: str = "1h",
    db_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Full offline evaluation: Hyper AI pipeline → StrategySpec → evaluate_spec_individual.
    Returns the evaluation result dict (fitness, gate, metrics, etc.).
    """
    from backtest.strategy_spec import evaluate_spec_individual

    # Run Hyper AI pipeline
    strategy_code, trace = run_hyper_ai_offline(
        user_request, exchange, symbols, db_factory
    )

    # Convert to StrategySpec
    spec_dict = strategy_code_to_spec(strategy_code)
    trace["derived_spec"] = spec_dict

    # Evaluate
    eval_result = evaluate_spec_individual(
        spec_dict,
        symbols or ["BTC"],
        exchange=exchange,
        backtest_days=backtest_days,
        end_time_ms=end_time_ms,
        kline_period=kline_period,
        generation=0,
        total_generations=1,
    )

    return {
        "strategy_code": strategy_code,
        "spec": spec_dict,
        "trace": trace,
        "evaluation": eval_result,
    }


if __name__ == "__main__":
    # Quick smoke test
    import time
    from datetime import datetime, timezone

    req = "EMA cross + Donchian breakout + RSI band on BTC 1h"
    end_ms = int(datetime(2026, 10, 31, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)

    print(f"Testing Hyper AI offline pipeline: {req}")
    t0 = time.time()
    try:
        result = evaluate_hyper_ai_output(
            req, "hyperliquid", ["BTC"], backtest_days=180, end_time_ms=end_ms
        )
        ev = result["evaluation"]
        print(f"Done in {time.time()-t0:.1f}s")
        print(f"  fitness: {ev.get('fitness')}")
        print(f"  gate_annual: {ev.get('gate_annual')}")
        print(f"  pnl_365d: {ev.get('pnl_365d')}")
        print(f"  maxdd_365d: {ev.get('maxdd_365d')}")
        print(f"  sharpe_365d: {ev.get('sharpe_365d')}")
        print(f"  pf: {ev.get('pf')}")
        print(f"  trades_365d: {ev.get('trades_365d')}")
        print(f"  win_rate: {ev.get('win_rate')}")
        print(f"  spec_id: {ev.get('spec_id')}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

def evaluate_strategy_code(
    strategy_code: str,
    symbols: Optional[List[str]] = None,
    exchange: str = "hyperliquid",
    generation: int = 0,
    total_generations: int = 100,
    backtest_days: int = 180,
    end_time_ms: Optional[int] = None,
    kline_period: str = "1h",
    db_factory: Optional[Any] = None,
    max_triggers: int = 8000,
) -> Dict[str, Any]:
    """
    Evaluate raw strategy code directly using the same path as evaluate_individual.
    This bypasses StrategySpec -> render, works with any valid sandbox strategy.
    """
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

    from backtest.ab_validators import annual_validator, exploration_validator
    from backtest.ga_annual import (
        EXPLORATION_GENS_OVERRIDE,
        calculate_fitness,
        calculate_exploration_fitness,
        gate_violation,
    )
    from database.connection import SessionLocal

    override = EXPLORATION_GENS_OVERRIDE
    if override is not None and override > 0:
        exploration_gens = int(override)
    else:
        exploration_gens = max(3, int(total_generations * 0.5))
    exploration_gens = min(exploration_gens, max(1, total_generations))
    use_exploration = (generation - 1) < exploration_gens

    validate = exploration_validator if use_exploration else annual_validator
    calc = calculate_exploration_fitness if use_exploration else calculate_fitness

    db = (db_factory() if db_factory else SessionLocal())
    try:
        ga, info = validate(
            db, strategy_code, symbols=symbols or ["BTC"], exchange=exchange,
            fast_fail=True, max_triggers=max_triggers,
            backtest_days=backtest_days, end_time_ms=end_time_ms
        )
    except Exception as e:
        return {
            "gate_annual": False,
            "pnl_365d": 0.0,
            "maxdd_365d": 100.0,
            "sharpe_365d": 0.0,
            "pf": None,
            "trades_365d": 0,
            "win_rate": 0.0,
            "fitness": -1000.0,
            "error": str(e)[:500],
        }
    finally:
        try:
            db.close()
        except Exception:
            pass

    f365 = info.get("f365", {}) if info else {}
    pnl = float(f365.get("pnl_percent", 0) or 0)
    dd = float(f365.get("maxdd_pct", 100) or 100)
    sharpe = float(f365.get("sharpe", 0) or 0)
    pf = f365.get("pf")
    pf_f = float(pf) if pf is not None else None
    trades = int(f365.get("trades", 0) or 0)
    win = float(f365.get("win", 0) or 0)

    # 從 strategy_code 解析 portion/leverage，用於 gate_violation 的 deploy 項。
    # 與 EvalResult 同名同義（genome.portion / genome.leverage），讓 fitness
    # 與 spec 路徑（evaluate_spec_individual）逐位一致。
    code_portion, code_leverage = _extract_risk_from_code(strategy_code)

    class DummyGenome:
        def __init__(self):
            self.portion = code_portion
            self.leverage = code_leverage

    class DummyResult:
        def __init__(self):
            self.genome = DummyGenome()
            self.pnl_365d = pnl
            self.maxdd_365d = dd
            self.sharpe_365d = sharpe
            self.pf = pf_f
            self.trades_365d = trades
            self.win_rate = win

    res = DummyResult()
    try:
        fitness = float(calc(res))
    except Exception:
        fitness = -1000.0

    return {
        "gate_annual": bool(ga),
        "pnl_365d": pnl,
        "maxdd_365d": dd,
        "sharpe_365d": sharpe,
        "pf": pf_f,
        "pf_ok": info.get("pf_ok", False) if info else False,
        "trades_365d": trades,
        "win_rate": win,
        "fitness": fitness,
        "spec_id": None,
        "f365": f365,
        "info": info,
        "risk": {"portion": code_portion, "leverage": code_leverage},
    }


_DECISION_CALL_RE = re.compile(r"Decision\s*\((.*?)\)\s*,?\s*$", re.DOTALL | re.MULTILINE)
_PORTION_RE = re.compile(r"target_portion_of_balance\s*=\s*([\d.]+)")
_LEVERAGE_RE = re.compile(r"leverage\s*=\s*(\d+)")
_OP_RE = re.compile(r'operation\s*=\s*["\'](buy|sell|close|hold)["\']')


def _extract_risk_from_code(code: str) -> tuple:
    """從 strategy_code 解析 **進場** portion/leverage，供 gate_violation deploy 項。

    只看 operation="buy"/"sell" 的 Decision（close 的 target_portion_of_balance=1.0
    是「平掉整個倉位」而非開倉規模，把它算進去會誤判 deploy 違規）。
    多個進場分支取最大值（最壞情況名目）。
    找不到時 fail-open 回預設 0.35 / 2（與 render_strategy_from_spec 一致）。
    """
    portions: List[float] = []
    leverages: List[int] = []
    # 以 "Decision(" 切段，逐段檢查 operation，抓同一段內的參數
    for chunk in code.split("Decision")[1:]:
        head = chunk[:600]  # Decision 參數不會超過此長度
        op_m = _OP_RE.search(head)
        if not op_m or op_m.group(1) not in ("buy", "sell"):
            continue
        for m in _PORTION_RE.finditer(head):
            try:
                v = float(m.group(1))
                if 0 < v <= 1.0:
                    portions.append(v)
            except ValueError:
                pass
        for m in _LEVERAGE_RE.finditer(head):
            try:
                v = int(m.group(1))
                if 1 <= v <= 50:
                    leverages.append(v)
            except ValueError:
                pass
    portion = max(portions) if portions else 0.35
    leverage = max(leverages) if leverages else 2
    return portion, leverage

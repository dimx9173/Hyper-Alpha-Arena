# Handoff: Hyper-Alpha-Arena → minipc (2026-10-30)

本機 `/home/brian/project/Hyper-Alpha-Arena` 已刪除。
唯一工作區：`~/Projects/Hyper-Alpha-Arena`，branch `fork/brian`，remote `origin/fork/brian`。

## 專案一句話

Hyperliquid + Binance crypto perps 平台。單模板 GA
（`backend/backtest/ga_annual.py` 19 基因：EMA + Donchian + RSI + ICT OTE）
→ `render_strategy` 程式碼生成 → `program_execution_service.py` 實盤。

## 護欄（策略請求違反即 skip，不算失敗）

- 總暴露 ≤2x，單幣 ≤0.4，同簇 ≤0.6（BTC/ETH/SOL/HYPE/BNB/AVAX/NEAR）
- 日虧 -5%，同向加倉冷卻 4h，最多 3 併發
- TP limit / SL **MARKET**，IOC→GTC→90s TTL

## 已知問題（按修復優先級）

1. **費率分裂**：`backend/services/paper_broker.py:54-95`
   （1.5bp/4.5bp/2bp）vs `backend/backtest/execution_simulator.py:33-40`
   （0.035%/0.02%/0.05%）→ paper 驗證失真。修：統一單一來源。
2. **疑似前視**：`backend/services/signal_backtest_service.py:502-508`
   （raw_data 未按 check_time 過濾）；`backend/backtest/historical_data_provider.py:289-295`
   （virtual kline 注入未來 mark_price）。修：加合成前視測試。
3. **GA 與護欄脫鉤**：`portion 0.45 × leverage 3 = 1.5x` 回測高分、
   實盤被 `single_coin_cap` 全擋（曾 7 天零成交）。修：fitness 內建硬拒。
4. **針峰**（`STRATEGY_CORE.md` §四之二）：iter10 R7 年化 +22.53%/Sharpe 2.135，
   鄰域 ±5% 擾動中位 -17.55%。`TRADE_COUNT_MIN=60` 已加但未根治。

## 測試（`cd backend`）

```bash
.venv/bin/pytest test_paper_broker.py test_ict_smc.py -q          # 34 passed
.venv/bin/python backtest/test_max_triggers_units.py              # 23/23
.venv/bin/python backtest/test_engine_accounting.py               # 31/31
.venv/bin/python backtest/test_ga_scoring_regression.py          # 3 項失敗（已知，見下）
```

- `test_engine_accounting.py` 有頂層 `sys.exit`，**不能**進 pytest collection。
- `test_ga_scoring_regression.py` 3 失敗（fitness 公式/探索懲罰/交易數無關項）
  在 `de9dc32` 乾淨樹已存在：測試寫死舊 `GATE_PENALTY_W`/20 筆門檻，
  `ga_annual.py` 後來改了交易數懲罰。修測試或回公式，另開任務。
- 無 docker：DB/compose 整合測試跑不了，只跑純邏輯。

## 環境

- `backend/.venv`（Python 3.13，uv 管理）。`uv sync --dev` 可重建。
- `cold-data` → `~/pCloud Drive/brian-32gb-drive/Hyper-Alpha-Arena/backtest`
  （機器相關 symlink，已 gitignore，勿提交）。
- GA 產物（`best_code_annual_ga*.py`、`ga_*.meta.json`、`summary_annual_ga*.json`）
  已 gitignore，勿提交。

## Git 流程

改完測過 → `git add -A && git commit -m "..." && git push`（`fork/brian` 已設 upstream）。
本機已無專案，minipc 即真相。

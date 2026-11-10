# PLAN: Backtest Engine MTM 修復

## Tasks

### T1: 強制平倉核心邏輯
- [ ] 新增 `_force_close_all(account, prices, config, end_time_ms)` 函數
  - 遍歷 `account.positions` 所有持倉
  - 取該 symbol ≤ end_time 的最後一根已收盤 1h K 線價格
  - 呼叫既有平倉路徑（同一 fee/slippage 模型），不可手算
  - 標記 `exit_reason="end_of_backtest"`
  - 缺價 fallback 到最後可得價並打 warning flag
- [ ] 在 `_calculate_stats` 開頭（計算任何指標前）呼叫 `_force_close_all`

### T2: Equity Curve 重取樣與 Sharpe 年化修正
- [ ] 新增 `_resample_equity_daily(equity_curve, start_ms, end_ms)` 函數
  - 輸入：trigger 時刻的 equity 點（間隔不規則）
  - 輸出：固定日曆網格（每日 00:00 UTC）的 equity 值，線性插值
  - 端點：start_ms 對齊到當日 00:00，end_ms 對齊到當日 00:00
- [ ] 修改 Sharpe 計算區塊（約 L779-795）
  - 取重取樣後的日頻 equity 計算日報酬
  - 年化因子改為 `√365`（crypto 全年無休）
  - 保留現有 guard（σ=0、單點、prev_eq≤0）

### T3: Funding 費用入帳
- [ ] 在 `_calculate_stats` 開頭將 `account.total_funding_shadow` 扣入 `account.balance`
- [ ] 將 `account.total_funding_shadow` 歸零
- [ ] 或採替代方案：明文排除並在 BacktestResult 加欄位揭露（二選一，建議入帳）

### T4: 會計恆等式斷言
- [ ] 在 `_calculate_stats` 結束前斷言：
  ```python
  assert abs((account.equity - config.initial_balance) - sum(t.pnl for t in closed_trades)) < 1e-6
  ```
- [ ] 失敗時拋出具體錯誤而非 assert（生產環境友善）

### T5: 回歸測試更新與新增測試
- [ ] 更新既有測試的 expected values（因強平會改變 PnL、trades、PF、勝率）
- [ ] 新增單元測試（見 Test Plan）：
  - test_force_close_all_basic
  - test_force_close_all_multiple
  - test_force_close_all_no_position
  - test_force_close_all_missing_price
  - test_resample_equity_daily
  - test_sharpe_annualization
  - test_accounting_identity
  - test_funding_in_equity
- [ ] 新增整合測試：
  - test_btc_buy_and_hold_benchmark
  - test_full_backtest_metrics_consistency

### T6: 基準驗證與重跑
- [ ] 跑 BTC 買入持有 365 天基準，驗證指標與獨立計算一致
- [ ] 用修復後引擎重跑 12 候選 × 3 區間，確認 Y2 盯市報酬 ≤ 20%
- [ ] 跑完整 132 項回歸測試

## Execution Order
T1 → T2 → T3 → T4 → T5 → T6

## Dependencies
- T1 必須先完成（T2-T4 依賴 closed_trades 已含強平交易）
- T5 可與 T1-T4 平行寫測試，但執行需等 T1-T4 完成
- T6 需 T1-T5 全部綠燈

## Estimated Effort
- T1: 2-3 hrs（核心邏輯，需仔細對齊既有平倉路徑）
- T2: 1-2 hrs（重取樣邏輯 + Sharpe 修正）
- T3: 0.5 hr
- T4: 0.5 hr
- T5: 2-3 hrs（測試較多）
- T6: 1-2 hrs（含等待回測跑完）

Total: ~8-11 hrs
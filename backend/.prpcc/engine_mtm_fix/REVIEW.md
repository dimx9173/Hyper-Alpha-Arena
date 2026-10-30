# REVIEW: Backtest Engine MTM 修復

## Review Checklist

### Correctness
- [ ] 強制平倉邏輯與既有平倉路徑完全一致（同一 fee/slippage 模型）
- [ ] 所有未平倉部位都被平倉，無遺漏
- [ ] 缺價 fallback 機制正確，不靜默跳過
- [ ] 強平交易正確計入 total_trades、total_pnl、PF、win_rate
- [ ] Equity curve 重取樣到日頻網格正確（線性插值、端點對齊）
- [ ] Sharpe 年化因子 `√365` 正確，守衛條件保留
- [ ] Funding 正確扣入 balance/equity
- [ ] 會計恆等式斷言通過

### Spec Compliance
- [ ] 不改變任何 gate 門檻數值
- [ ] 歷史分數標記「engine v1 口徑，作廢」
- [ ] BacktestResult 帶 engine 版本號

### Performance
- [ ] 執行時間增加 < 5%
- [ ] 無額外記憶體洩漏

### Security
- [ ] 無注入風險
- [ ] 無秘密洩漏

### Tests
- [ ] 所有 132 項回歸測試通過
- [ ] 新增單元測試覆蓋 T1-T4 所有邊界情況
- [ ] 整合測試覆蓋 BTC 基準與完整回測一致性

## Verdict
- [ ] PASS
- [ ] FAIL
- [ ] CONDITIONAL (列出 blocking issues)

## Notes
- Codex 僅審查，不寫代碼
- 發現問題請具體到行號、函數名、預期 vs 實際行為

---

**Reviewer (Codex)**: 待簽核
**Date**: 待填入
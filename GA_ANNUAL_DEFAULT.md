# GA + Annual (365d) 為預設策略迭代模式

## 概述

自 v1.1.0 起，**基因演算法 (GA) + 年度回測 (365+ 天)** 成為 Hyper-Alpha-Arena 的**預設且唯一**策略迭代模式。

### 核心原則

| 項目 | 規定 |
|------|------|
| **優化引擎** | 基因演算法 (GA) — 僅支援 GA，移除單親代爬山法 |
| **回測週期** | **強制 365 天以上** — 不接受更短週期 |
| **驗證標準** | Annual Gate (365d validator) — 月均 10%、Max DD ≤ 5% |
| **輸出產物** | `best_code_annual.py` + `summary_annual.json` |

---

## 快速開始

```bash
# 進入專案目錄
cd /home/brian/project/Hyper-Alpha-Arena

# 執行預設 GA 年度迭代（25 代、族群 20、4 workers）
PYTHONPATH=backend python -m backtest.iterate

# 自訂參數
PYTHONPATH=backend python -m backtest.iterate \
    --generations 30 \
    --pop-size 30 \
    --workers 8 \
    --early-stop 10

# 續跑
PYTHONPATH=backend python -m backtest.iterate --resume
```

---

## 參數說明

### GA 核心參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--generations` | 25 | 進化代數 |
| `--pop-size` | 20 | 族群大小 |
| `--elite` | 2 | 精英保留數 |
| `--tournament` | 3 | 錦標賽大小 |
| `--cx-rate` | 0.7 | 交配率 |
| `--mut-rate` | 0.2 | 突變率 |
| `--early-stop` | 8 | 連續無改善早停代數 |

### 年度回測參數（強制）
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--min-days` | **365 (不可降低)** | 最少回測天數，強制年度級 |
| `--target-pnl-pct` | 120.0 | 目標年化報酬 % (月均 10%) |
| `--max-maxdd-pct` | 5.0 | 最大回撤 % 上限 |

### 其他參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--exchange` | hyperliquid | 交易所 |
| `--symbols` | HL Top30 | 指定幣種 |
| `--seed` | None | 隨機種子 |
| `--output-dir` | . | 輸出目錄 |
| `--no-validate` | False | 跳過驗證（不建議） |

---

## Annual Gate 驗證標準 (ab_validators.annual_validator)

策略必須同時通過以下所有條件才視為 **PASS**：

| 指標 | 門檻 | 說明 |
|------|------|------|
| 年化報酬 (PnL%) | **≥ 120%** | 月均 10% |
| 最大回撤 (MaxDD%) | **≤ 5%** | 極嚴格風控 |
| 獲利因子 (PF) | **> 1.5** | 盈虧比 |
| 年化 Sharpe | **> 0.5** | 風險調整報酬 |
| 勝率 (Win%) | **> 45%** | 交易勝率 |
| 年交易次數 | **≥ 50** | 確保樣本數 |
| 5m K線日均 | **≥ 50** | 數據完整度 |

> **注意**: 任一項未達標即判定 **FAIL**，不計入候選策略。

---

## 基因組參數空間

```python
GENOME_BOUNDS = {
    # EMA 趨勢
    "ema_fast": (10, 30),
    "ema_slow": (40, 80),
    "ema_tol_buy": (0.995, 0.9995),
    "ema_tol_sell": (1.0005, 1.002),
    
    # Donchian 突破
    "donchian_thr": (0.97, 0.995),
    
    # RSI 平均回歸
    "rsi_buy_thr": (25, 45),
    "rsi_sell_thr": (55, 75),
    "use_rsi": [True, False],
    
    # 風險管理
    "tp_pct": (0.04, 0.12),    # 4-12%
    "sl_pct": (0.02, 0.08),    # 2-8%
    "portion": (0.15, 0.5),    # 15-50% 倉位
    "leverage": (1, 3),        # 1-3x 槓桿
}
```

---

## 輸出檔案結構

```
backend/backtest/
├── iterate.py                    # 統一入口 (GA + Annual 預設)
├── iterate_annual_ga.py          # GA 年度迭代實作
├── ga_annual.py                  # GA 引擎核心
├── ab_validators.py              # Annual Gate 驗證器
├── best_code_annual_ga.py        # 最終最佳策略代碼
├── summary_annual_ga.json        # 年度回測摘要
├── iteration_log_annual_ga.jsonl # 迭代日誌 (每代一筆)
```

### summary_annual_ga.json 範例

```json
{
  "best_fitness": 1.847,
  "best_generation": 17,
  "generations": 25,
  "population_size": 20,
  "best_genome": {
    "ema_fast": 22,
    "ema_slow": 58,
    "ema_tol_buy": 0.9982,
    "ema_tol_sell": 1.0011,
    "donchian_thr": 0.983,
    "rsi_buy_thr": 32,
    "rsi_sell_thr": 68,
    "use_rsi": true,
    "tp_pct": 0.085,
    "sl_pct": 0.045,
    "portion": 0.28,
    "leverage": 2
  },
  "backtest_365d": {
    "pnl_pct": 142.3,
    "maxdd_pct": 4.2,
    "sharpe": 0.87,
    "profit_factor": 1.78,
    "win_rate": 52.1,
    "trades": 67,
    "gate_annual": true
  }
}
```

---

## 為什麼強制 365 天？

1. **統計顯著性**: 短週期回測極易過擬合，365 天提供足夠樣本數 (≥50 筆交易)
2. **市場機制完整性**: 涵蓋牛市、熊市、震盪、高波動、低波動等完整市場循環
3. **風險指標可信度**: MaxDD、Sharpe、PF 在短週期極不穩定，年度級才具參考價值
4. **實戰對齊**: 實盤策略需跨越季度、財報季、宏觀事件，年度回測最貼近實戰

---

## 遷移指南

### 從舊版 iterate.py 遷移

| 舊指令 | 新指令 |
|--------|--------|
| `python -m backtest.iterate --rounds 30` | `python -m backtest.iterate --generations 25` |
| `python -m backtest.iterate_annual.py` | `python -m backtest.iterate` (預設即為 GA+Annual) |
| 手動調參 | GA 自動搜尋最佳參數組合 |

### 重要變更

1. **不再支援** `--rounds` 參數，改用 `--generations`
2. **不再支援** 短週期回測 (30d, 90d 等)，強制 365d+
3. **不再支援** 單親代突變模式，改為完整 GA (選擇+交配+突變+精英)
4. **驗證標準** 統一為 Annual Gate，移除 Gate A/B/C 分階段

---

## 故障排除

### 迭代速度過慢
- 減少 `--workers` (預設 4，視 CPU 核心調整)
- 檢查資料庫連線是否為瓶頸
- 確認 `--min-days` 未被意外修改

### 找不到通過策略
- 增加 `--generations` 到 50
- 增加 `--pop-size` 到 30-40
- 檢查 `iteration_log_annual_ga.jsonl` 中的適應度分佈
- 確認歷史資料完整性 (5m K線日均 ≥ 50)

### 早熟收斂
- 增加 `--mut-rate` 到 0.3
- 減少 `--elite` 到 1
- 使用 `--seed` 重新開始

---

## 相關檔案

- `backend/backtest/iterate.py` — 統一入口
- `backend/backtest/iterate_annual_ga.py` — GA 年度迭代 CLI
- `backend/backtest/ga_annual.py` — GA 引擎
- `backend/backtest/ab_validators.py` — Annual Validator
- `.omo/PRP_ANNUAL_GA.md` — 產品需求文件
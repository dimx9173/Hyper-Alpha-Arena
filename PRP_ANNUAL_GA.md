# PRP: Annual Genetic Algorithm Strategy Optimizer

## 目標
將現有單親代爬山法重構為真正的基因演算法（GA），針對年化回測（365d）優化策略，達成：
- 月均 PnL ≥ 10%（年化 ≥ 120%）
- Max DD ≤ 5%
- PF > 1.5, Sharpe > 0.5, Win > 45%, Trades ≥ 50/年

---

## 規格

### 1. 族群架構
- **族群大小**：20 個體
- **精英保留**：前 2 名直接進入下一代
- **選擇機制**：錦標賽選擇（tournament size=3）
- **交配率**：0.8
- **突變率**：0.2（每基因）
- **代數**：25 代（20×25=500 評估，對應原 100 輪預算）

### 2. 基因型編碼
每個體基因組 = 策略參數字典：
```python
{
    "ema_fast": 20,           # EMA 快線週期 10-30
    "ema_slow": 50,           # EMA 慢線週期 40-80
    "ema_tol_buy": 0.9985,    # 金叉確認容差 0.995-0.9995
    "ema_tol_sell": 1.0008,   # 死叉確認容差 1.0005-1.002
    "donchian_thr": 0.98,     # Donchian 突破閾值 0.97-0.995
    "rsi_period": 14,         # RSI 週期（固定）
    "rsi_buy_thr": 35,        # RSI 買入閾值 25-45
    "rsi_sell_thr": 65,       # RSI 賣出閾值 55-75
    "use_rsi": True,          # 是否啟用 RSI
    "use_volume": False,      # 是否啟用成交量（VOLUME 指標不存在）
    "vol_mult": 1.2,          # 成交量倍數 1.0-2.0
    "tp_pct": 0.08,           # TP 4-12%
    "sl_pct": 0.04,           # SL 2-8%
    "portion": 0.25,          # 倉位比例 0.15-0.5
    "leverage": 2,            # 槓桿 1-3
}
```

### 3. 適應度函數
```python
def fitness(result):
    if not result.gate_annual:
        return -1000  # 硬性懲罰
    # 多目標：PnL 權重 0.4, -DD 權重 0.3, Sharpe 權重 0.2, PF 權重 0.1
    return (
        0.4 * min(result.pnl_365d / 120, 2.0) +   # PnL 上限 240% 得滿分
        0.3 * max(0, 1 - result.maxdd_365d / 5) + # DD ≤5% 得滿分
        0.2 * min(result.sharpe_365d / 1.0, 2.0) + # Sharpe ≥1 得滿分
        0.1 * min(float(result.pf or 0) / 2.0, 2.0) # PF ≥2 得滿分
    )
```

### 4. 交配操作（參數級混合）
```python
def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if isinstance(parent1[key], (int, float)):
            # 混合交配
            alpha = random.random()
            child[key] = parent1[key] * alpha + parent2[key] * (1 - alpha)
            # 整數參數取整
            if isinstance(parent1[key], int):
                child[key] = int(round(child[key]))
        else:
            # 布林/類別：隨機選擇
            child[key] = random.choice([parent1[key], parent2[key]])
    return clamp(child)
```

### 5. 突變操作（高斯擾動 + 結構變異）
```python
def mutate(genome):
    for key in genome:
        if random.random() < 0.2:  # 每基因 20% 突變率
            if key in ["tp_pct", "sl_pct", "portion", "ema_tol_buy", "ema_tol_sell", "donchian_thr"]:
                # 高斯擾動 ±10%
                genome[key] *= (1 + random.gauss(0, 0.1))
            elif key in ["ema_fast", "ema_slow", "rsi_buy_thr", "rsi_sell_thr"]:
                # 整數步進 ±2
                genome[key] += random.choice([-2, -1, 1, 2])
            elif key in ["use_rsi", "use_volume", "leverage"]:
                # 離散跳躍
                genome[key] = random.choice([1, 2, 3]) if key == "leverage" else not genome[key]
    return clamp(genome)
```

### 6. 邊界約束
```python
BOUNDS = {
    "ema_fast": (10, 30), "ema_slow": (40, 80),
    "ema_tol_buy": (0.995, 0.9995), "ema_tol_sell": (1.0005, 1.002),
    "donchian_thr": (0.97, 0.995),
    "rsi_buy_thr": (25, 45), "rsi_sell_thr": (55, 75),
    "vol_mult": (1.0, 2.0),
    "tp_pct": (0.04, 0.12), "sl_pct": (0.02, 0.08),
    "portion": (0.15, 0.5), "leverage": (1, 3),
}
```

### 7. 執行流程
```
初始化：從 baseline 提取基因 → 生成 20 個體（含 baseline + 隨機擾動）
For generation in 1..25:
    並行評估：每個體跑 annual_validator（可用 ProcessPoolExecutor max_workers=4）
    計算適應度
    精英保留：前 2 直接複製
    選擇 + 交配 + 突變 → 產生 18 新個體
    新族群 = 精英 + 子代
    記錄：最佳適應度、族群多樣性、通過率
    若連續 5 代無改善 → 早停
輸出：最佳個體基因 → 渲染策略代碼 → 寫入 best_code_annual_ga.py
```

### 8. 輸出產物
- `backend/backtest/ga_annual.py` - GA 引擎
- `backend/backtest/iterate_annual_ga.py` - CLI 入口
- `backend/backtest/iteration_log_annual_ga.jsonl` - 每代記錄
- `backend/backtest/best_code_annual_ga.py` - 最終最佳策略

---

## 驗收標準
- [ ] GA 引擎獨立可測試（單元測試覆蓋選擇/交配/突變/邊界）
- [ ] 25 代內找到 ≥1 個通過 annual_validator 的個體
- [ ] 最佳個體：PnL ≥120%, DD ≤5%, Sharpe >0.5
- [ ] 族群多樣性不提前收斂（基因型熵 > 閾值）
- [ ] 總執行時間 < 6 小時（並行 4 workers）

---

## 風險與緩解
| 風險 | 緩解 |
|------|------|
| 回測耗時過長（35040 triggers × 500 = 17.5M） | 並行 4 workers、早停、快取數據 |
| 早熟收斂 | 突變率自適應、多樣性注入 |
| VOLUME 指標不存在 | 移除 use_volume 基因，改用 ATR/動量 |
| 參數空間過大 | 先跑敏感度分析縮小範圍 |

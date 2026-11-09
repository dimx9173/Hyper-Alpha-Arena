# 「伏羲動態滾動量化系統」 (Fuxi-Adaptive Rolling Quant)   
機器學習量化交易系統架構規格書 (Production-Grade Quantitative Trading Specification v4.0)

本規格書定義一套針對加密貨幣永續合約（Perpetual Futures）短線週期（5m, 15m, 30m）的機構級\*\*全自動滾動閉環（Rolling Closed-Loop）\*\*機器學習量化交易系統。本架構採用 **兩階段元標籤架構（Meta-Labeling）** 解耦方向預測與部位管理，結合 **跨週期無洩漏特徵融合（Point-in-Time Multi-Timeframe Alignment）** 與 **K線微結構/技術指標定常化**；在模型訓練層引入 **三重障礙法（Triple Barrier Method）**、**樣本唯一性權重（Sample Uniqueness Weighting）** 與 **混合抽樣抗遺忘視窗（Hybrid Anti-Amnesia Window）**；在滾動生命週期層設計 **三層異步滾動循環（Multi-Cycle Asynchronous Rolling）**、**衛冕者-挑戰者無縫交接（Champion-Challenger Gate）** 與 **孤兒持倉隔離（Orphan Position Isolation）**；在資金與風控層整合 **非對稱階梯式棘輪複利與安全金庫（Asymmetric Compounding Ratchet &amp; Reserve Vault）**、**合約強平邊界防禦（Liquidation Guard）** 與 **月度波動率目標及高水位鎖利機制（Monthly Equity Smoothing &amp; HWM Skimming）**；最後透過 **四大收益驗證關卡（DSR 降維夏普檢驗、蒙地卡羅路徑重抽樣、摩擦敏感度壓力測試、月度體感指標）** 確保系統在7x24小時實盤滾動中具備統計顯著且抗脆弱的正期望值（Edge）。

---

## 1. 系統全域與三層異步滾動架構 (End-to-End &amp; Multi-Cycle Rolling Architecture)

```
[ 多週期歷史/即時數據源: 5m (主軸), 15m, 30m OHLCV + Funding Rate + Open Interest ]
                                        │
       ┌────────────────────────────────┼────────────────────────────────┐
       ▼ (Loop 1: 每 5m 即時推論)        ▼ (Loop 2: 每 7~14天 權重滾動)    ▼ (Loop 3: 每 30天/PSI觸發 結構重構)
┌──────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
│ 高頻推論與風控執行循環     │  │ 中頻模型權重滾動重訓循環    │  │ 低頻結構與超參數高原循環    │
│ 1. 跨週期無洩漏特徵更新    │  │ 1. 70%近期 + 30%壓力記憶庫 │  │ 1. 48指標去共線性與SHAP篩選 │
│ 2. 衛冕者模型推論 p_cal   │  │ 2. 樣本唯一性權重重新計算  │  │ 2. (N,M,x,y) 參數高原搜尋  │
│ 3. 強平防禦 + 棘輪凱利注碼 │  │ 3. 重訓 LGBM 與機率校準層  │  │ 3. 報復性過擬合檢測 (Pass) │
└─────────────┬────────────┘  └─────────────┬─────────────┘  └─────────────┬─────────────┘
              │                             │                              │
              │                             └──────────────┬───────────────┘
              │                                            ▼
              │                 ┌──────────────────────────────────────────────────┐
              │                 │ 衛冕者 vs. 挑戰者驗證與交接閘門 (Handoff Gate)     │
              │                 │ ├─ 四大關卡檢驗：DSR>0.95 + MC 95% MDD<=20%       │
              │                 │ ├─ 影子盤對決：Challenger 樣本外期望值 > Champion │
              │                 │ └─ 孤兒持倉隔離：既有部位鎖定原模型(U_t, D_t, M)  │
              │                 └──────────────────────────┬───────────────────────┘
              ▼                                            │ 晉升新衛冕者 (Active Champion)
┌──────────────────────────────────────────────────────────┴───────────────────────┐
│ 實盤持倉狀態機、API 對帳與階梯式棘輪資金池 (State Machine & Compounding Ratchet)   │
│ ├─ API 狀態對帳 (Reconciliation)：每 5m 對齊交易所實際持倉與本地狀態機            │
│ ├─ 月度鎖利與防禦 (Monthly HWM & DD Throttle)：達標減半保果實、觸底熔斷休眠       │
│ └─ 非對稱棘輪金庫 (Asymmetric Ratchet)：週期結算抽水 (1-α) 入冷金庫，鎖死資產底線 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 跨週期無洩漏對齊與定常化特徵工程

### 2.1 跨時間週期無洩漏對齊 (Point-in-Time Multi-Timeframe Alignment)

以最小交易週期 $5\text{m}$ 作為主時間軸 $t$。設高階週期為 $H \in \{15\text{m}, 30\text{m}\}$：

- **嚴格收盤約束（Closed-Bar Constraint）**：在主時間軸時刻 $t$，高階週期 $H$ 的最新可用索引為 $\tau_H(t)$，必須滿足該高階 K 線的結束時間 $\text{CloseTime}(\tau_H) \le t$。
- **冷啟動預熱緩衝（Warm-Start Feature Buffer）**：每次模型滾動切換或容器重啟時，必須預先載入至少 $\max(N, 60) \times \frac{30\text{m}}{5\text{m}} = 360$ 根歷史 $5\text{m}$ K 線進行指標預熱快取，確保第 1 根即時推論無 `NaN` 或平滑失真。

### 2.2 K線微結構幾何特徵 (Price Action Geometry)

給定週期 $H$ 下第 $t$ 根 K 線之開高低收量 $(O_t, H_t, L_t, C_t, V_t)$，極小值 $\epsilon = 10^{-8}$：

1. **實體漲跌幅（Body Return）**：
   $$
   F_{\text{body}, t} = \frac{C_t - O_t}{O_t + \epsilon}
   $$
2. **上下影線佔比（Shadow Ratios）**：
   $$
   F_{\text{upper}, t} = \frac{H_t - \max(O_t, C_t)}{H_t - L_t + \epsilon}, \quad F_{\text{lower}, t} = \frac{\min(O_t, C_t) - L_t}{H_t - L_t + \epsilon}
   $$
3. **相對振幅與滯後對數報酬（Normalized Range &amp; Lagged Log-Returns）**：
   $$
   F_{\text{range}, t} = \frac{H_t - L_t}{C_t + \epsilon}, \quad r_{t-k} = \ln\left(\frac{C_{t-k}}{C_{t-k-1}}\right) \quad (k \in \{0, \dots, N-1\})
   $$
4. **相對成交量（Relative Volume, RVOL）**：
   $$
   \text{RVOL}_t = \frac{V_t}{\frac{1}{20}\sum_{i=0}^{19} V_{t-i} + \epsilon}
   $$

### 2.3 定常化技術指標與合約微結構特徵 (Stationary Indicators &amp; Crypto Microstructure)

1. **均線偏離度（EMA Bias）**：
   $$
   F_{\text{bias}, k, t} = \frac{C_t - \text{EMA}_k(C)_t}{\text{EMA}_k(C)_t + \epsilon} \quad (k \in \{20, 60\})
   $$
2. **中心化強弱指標（Centered RSI）**：
   $$
   F_{\text{RSI}, t} = \frac{\text{RSI}_{14, t} - 50}{50} \in [-1, 1]
   $$
3. **相對波動度與布林帶特徵（Normalized ATR &amp; Bollinger Features）**：
   $$
   F_{\text{ATR}, t} = \frac{\text{ATR}_{14, t}}{C_t + \epsilon}, \quad \%B_t = \frac{C_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t + \epsilon}, \quad \text{BW}_t = \frac{\text{Upper}_t - \text{Lower}_t}{\text{SMA}_{20}(C)_t + \epsilon}
   $$
4. **資金費率與未平倉量動能（Funding Rate Z-Score &amp; OI Delta）**：
   $$
   F_{\text{funding}, t} = \frac{\text{FR}_t - \mu_{\text{FR}, 30}}{\sigma_{\text{FR}, 30} + \epsilon}, \quad F_{\text{OI}, t} = \ln\left(\frac{\text{OI}_t}{\text{OI}_{t-1} + \epsilon}\right)
   $$

---

## 3. 兩階段元標籤架構與樣本唯一性權重

### 3.1 第一階段：方向預測與三重障礙定義 (Primary Model &amp; Directional Triple Barrier)

1. **主模型（Primary Model）**：由高召回率規則（如跨週期動量突破、均線排列）給出方向預測 $d_t \in \{+1, -1\}$。
2. **方向性三重障礙（Directional Triple Barrier）**：給定進場價 $P_t$、目標止盈幅度 $x$、目標盈虧比 $y$（止損幅度為 $x/y$）、最大持有窗口 $M$ 根 K 線：
   - **停利邊界（Take-Profit Barrier）**：$B_{\text{TP}, t} = P_t \times (1 + d_t \cdot x)$
   - **停損邊界（Stop-Loss Barrier）**：$B_{\text{SL}, t} = P_t \times \left(1 - d_t \cdot \frac{x}{y}\right)$
   - **首觸時間（First-Passage Time）**：
     $$
     \tau_{\text{TP}} = \min\{\tau \in [1, M] \mid d_t \cdot (P_{t+\tau}^{\text{high/low}} - B_{\text{TP}, t}) \ge 0\}
     $$
     $$
     \tau_{\text{SL}} = \min\{\tau \in [1, M] \mid d_t \cdot (P_{t+\tau}^{\text{low/high}} - B_{\text{SL}, t}) \le 0\}
     $$
3. **第二階段元標籤（Meta-Label** $m_t \in \{0, 1\}$**）**：
   $$
   m_t = \begin{cases} 1 & \text{若 } \tau_{\text{TP}} < \tau_{\text{SL}} \text{ 且 } \tau_{\text{TP}} \le M \\ 0 & \text{其他情況（先觸及停損、同根K線雙向掃損，或到期未達標）} \end{cases}
   $$

    **Meta-Model（LightGBM）** 學習條件機率 $\hat{p}_t = P(m_t = 1 \mid X_t, d_t)$，並經 **Isotonic Regression** 校準為真實勝率 $\hat{p}_{\text{cal}, t}$。

### 3.2 樣本唯一性與權重分配 (Sample Uniqueness &amp; Concurrency Weighting)

1. **並行重疊計數（Concurrency** $c_\tau$**）**：
   $$
   c_\tau = \sum_{j=1}^{I} \mathbb{I}\left( \tau \in [t_{j, \text{start}}, t_{j, \text{end}}] \right)
   $$
2. **樣本平均唯一性與報酬加權（Return-Attributed Sample Weight** $w_i$**）**：
   $$
   u_i = \left( \frac{1}{t_{i, \text{end}} - t_{i, \text{start}} + 1} \sum_{\tau = t_{i, \text{start}}}^{t_{i, \text{end}}} \frac{1}{c_\tau} \right), \quad \tilde{w}_i = u_i \times \vert r_i \vert, \quad w_i = \tilde{w}_i \times \frac{I}{\sum_{j=1}^{I} \tilde{w}_j}
   $$

---

## 4. 三層異步滾動循環、抗遺忘視窗與無縫交接機制 (Rolling Closed-Loop Engine)

### 4.1 三層時間軸異步滾動定義 (Multi-Cycle Asynchronous Loops)


| 循環層級                | 執行頻率 / 觸發條件                  | 核心任務與凍結範圍                                                                                        | 防護目標                                  |
| :------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------ | :------------------------------------- |
| **Loop 1: 高頻推論與對帳** | 每根 5m K 線收盤                  | 執行交易所 API 狀態對帳（Reconciliation）、更新特徵快取、由當前衛冕者模型（Champion）計算 $\hat{p}_{\text{cal}, t}$ 並執行訂單與強平檢查。 | 確保零延遲執行並防止斷線導致本地與交易所部位脫節。             |
| **Loop 2: 中頻權重滾動**  | 每 7 天或 14 天定時                | **凍結特徵子集與超參數** $(N,M,x,y)$，利用「混合抽樣抗遺忘視窗」重新訓練 LightGBM 樹權重與 Isotonic 校準器，生成挑戰者模型（Challenger）。     | 適應近期市場微結構與波動尺度，避免交易風格劇烈擺盪。            |
| **Loop 3: 低頻結構重構**  | 每 30 天 或 $\text{PSI} > 0.25$ | 重新執行 48 指標去共線性與 SHAP 篩選，並透過高原目標函數 $J(\theta)$ 重新搜尋最佳超參數組合 $\theta^*$。                            | 應對宏觀市場狀態（Regime）切換與因子衰退（Alpha Decay）。 |


### 4.2 混合抽樣抗遺忘視窗 (Hybrid Anti-Amnesia Training Window)

為防止滑動視窗在經歷連續數週低波動橫盤後發生「市場記憶遺失（Regime Amnesia）」（即忘記極端暴跌或逼空特徵），每次滾動訓練集 $\mathcal{D}_{\text{train}}$ 由兩部分混合構成：

$$
\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{recent}}^{(1 - \rho)} \cup \mathcal{D}_{\text{stress}}^{(\rho)} \quad (\text{建議記憶混合比例 } \rho = 0.30)
$$

- **近期連續視窗** $\mathcal{D}_{\text{recent}}$**（佔** $70\%$**）**：取最近 $W_{\text{recent}}$ 天（如 60\~90 天）的連續樣本，捕捉最新盤口動能。
- **歷史極端壓力記憶庫** $\mathcal{D}_{\text{stress}}$**（佔** $30\%$**）**：自過去 1\~2 年歷史資料庫中，篩選相對波動率 $F_{\text{ATR}, t}$ 處於前 $5\%$ 分位數（閃崩、插針、單邊軋空）的獨立行情片段進行分層抽樣，強制模型始終保有對黑天鵝尾部風險的辨識力。
- **報復性過擬合熔斷（Revenge Overfitting Guard）**：在 Loop 3 搜尋超參數時，若所有候選參數高原的中位數夏普 $\max_{\theta} \mu_{\text{SR}}(\theta) < 0.5$，判定當前市場處於無趨勢噪音期，系統直接進入 **`PASS_REGIME`（全系統空倉休眠狀態）**，嚴禁強行挑出扭曲參數硬做。

### 4.3 衛冕者-挑戰者交接閘門與孤兒持倉隔離 (Champion-Challenger Gate &amp; Orphan Isolation)

1. **挑戰者晉升條件（Challenger Promotion Criterion）**：
 設線上衛冕者模型為 $\mathcal{M}_{\text{champ}}$，新訓練之挑戰者模型為 $\mathcal{M}_{\text{chal}}$。在滾動評估窗口（Walk-Forward OOS Window）與影子盤（Shadow Window）中，$\mathcal{M}_{\text{chal}}$ 必須同時滿足：
   $$
   \text{DSR}(\mathcal{M}_{\text{chal}}) \ge 0.95 \quad \land \quad \mathbb{E}[R_{\text{OOS}}(\mathcal{M}_{\text{chal}})] > \mathbb{E}[R_{\text{OOS}}(\mathcal{M}_{\text{champ}})] + \epsilon_{\text{margin}}
   $$

    若挑戰者未達標，則維持 $\mathcal{M}_{\text{champ}}$ 繼續服役，並將基礎凱利係數自動乘以衰減因子 $0.75$（防禦性降檔）。
2. **孤兒持倉隔離鐵律（Orphan Position Isolation）**：
 當 $\mathcal{M}_{\text{chal}}$ 成功晉升並替換 $\mathcal{M}_{\text{champ}}$ 時，若帳戶內仍存在由舊模型 $\mathcal{M}_{\text{champ}}$ 開立之未平倉部位 $\text{Pos}_{\text{old}}$，**嚴禁使用新模型的參數** $(x_{\text{new}}, y_{\text{new}}, M_{\text{new}})$ **修改該部位**。每一筆持倉在開倉瞬間即封裝其專屬出場契約包：
   $$
   \text{Contract}_k = \left\{ \text{ModelID}_k, \; B_{\text{TP}, k}, \; B_{\text{SL}, k}, \; t_{\text{expire}, k} \right\}
   $$

    舊部位必須嚴格依據其誕生時的 $\text{Contract}_k$ 獨立監控至平倉結算，新模型僅對交接完成後的新開倉位負責。

---

## 5. 合約強平防禦、階梯式棘輪複利與月度收益平滑機制

### 5.1 含資金費率與 Maker/Taker 之淨期望值 (Perpetual Net Expected Value)

設單邊手續費率為 $C_{\text{entry}}$、$C_{\text{exit}}$，滑價為 $S$，持有期間跨越資金費率結算點的預估資金成本為 $\Phi_t = d_t \cdot \text{FR}_t \cdot N_{\text{funding}}$：

$$
\text{Cost}_{\text{win}} = C_{\text{entry}} + C_{\text{maker}} + \Phi_t, \quad \text{Cost}_{\text{loss}} = C_{\text{entry}} + C_{\text{taker}} + S + \Phi_t
$$

$$
\mathbb{E}[R_t] = \hat{p}_{\text{cal}, t} \cdot \left(x - \text{Cost}_{\text{win}}\right) - \left(1 - \hat{p}_{\text{cal}, t}\right) \cdot \left(\frac{x}{y} + \text{Cost}_{\text{loss}}\right) > \delta_{\text{min}}
$$

### 5.2 強平邊界防禦約束 (Liquidation Boundary Guard)

設交易所維持保證金率為 $\text{MMR}$，安全緩衝墊為 $\Delta_{\text{buffer}}$。最大安全槓桿上限 $z_{\text{safe}}$ 為：

$$
\frac{x}{y} + \text{Cost}_{\text{loss}} + \Delta_{\text{buffer}} \le \frac{1}{z} - \text{MMR} \implies z \le z_{\text{safe}} = \left\lfloor \frac{1}{\frac{x}{y} + \text{Cost}_{\text{loss}} + \text{MMR} + \Delta_{\text{buffer}}} \right\rfloor
$$

### 5.3 非對稱階梯式棘輪複利與安全金庫 (Asymmetric Compounding Ratchet &amp; Reserve Vault)

為解決滾動系統中「連贏後在高點放大注碼、一次黑天鵝回撤把本利全吐光」的複利脆弱性，系統將總資產拆分為 **活躍交易資金池（Active Trading Pool,** $W_{\text{active}}$**）** 與 **不可動用安全金庫（Cold Reserve Vault,** $V_{\text{vault}}$**）**。

設初始總本金為 $W_0$（例如 $1000\text{u}$），棘輪結算週期為每週（或每 14 天滾動交接時）且處於空倉狀態，定義第 $k$ 個結算週期的已實現總淨值為 $W_{\text{realized}, k}$，歷史最高結算水位為 $\text{HWM}_{\text{ratchet}, k} = \max_{j \le k} W_{\text{realized}, j}$，利潤再投資率為 $\alpha \in [0.5, 0.7]$（即抽水比例為 $1 - \alpha$）：

1. **週期結算抽水入庫（Profit Skimming to Vault）**：
 當第 $k$ 期創造超越歷史結算高水位的新增淨利 $\Delta \Pi_k = \max\left(0, \; W_{\text{realized}, k} - \text{HWM}_{\text{ratchet}, k-1}\right)$ 時：
   $$
   V_{\text{vault}, k} = V_{\text{vault}, k-1} + (1 - \alpha) \cdot \Delta \Pi_k
   $$
2. **階梯式有效凱利基準本金（Ratcheted Base Capital** $W_{\text{base}, t}$**）**：
 在第 $k$ 期盤中時刻 $t$，若帳面回撤為 $\text{DD}_{\text{active}, t} = \max\left(0, \; W_{\text{active}, k} - W_{\text{current\_active}, t}\right)$，則用於計算凱利注碼的有效基準本金為：
   $$
   W_{\text{base}, t} = W_0 + \alpha \cdot \max\left(0, \; \text{HWM}_{\text{ratchet}, k} - W_0\right) - \text{DD}_{\text{active}, t}
   $$
   - **數學特性（非對稱棘輪）**：賺錢時僅在週期結算點以 $\alpha$ 比例階梯式上調注碼基準，其餘 $(1-\alpha)$ 永久鎖入安全金庫 $V_{\text{vault}}$ 絕不參與風險曝險；賠錢時則即時等額扣減 $W_{\text{base}, t}$ 迅速縮表防守，確保總資產底線隨時間單調遞增。

### 5.4 月度波動率目標與高水位鎖利機制 (Monthly Equity Smoothing &amp; HWM Skimming)

1. **波動率目標調節器（Volatility Targeting Scalar** $\eta_{\text{vol}, t}$**）**：
   $$
   \eta_{\text{vol}, t} = \text{clip}\left( \frac{\sigma_{\text{target}}}{F_{\text{ATR}, t} + \epsilon}, \; 0.5, \; 1.5 \right)
   $$
2. **月度高水位鎖利與回撤節流閥（Monthly HWM Profit Skimming &amp; Drawdown Throttle** $\eta_{\text{month}, t}$**）**：
 設當月累計收益率為 $R_{\text{MTD}, t}$，當月自高點回撤為 $\text{DD}_{\text{MTD}, t}$：
   $$
   \eta_{\text{month}, t} = \begin{cases} 0.0 \quad (\text{當月熔斷休眠}) & \text{若 } R_{\text{MTD}, t} \le -0.06 \text{ 或 } \text{DD}_{\text{MTD}, t} \ge 0.08 \\ 0.5 \quad (\text{逆風防守降檔}) & \text{若 } -0.06 < R_{\text{MTD}, t} \le -0.03 \\ 0.5 \quad (\text{達標鎖利模式}) & \text{若 } R_{\text{MTD}, t} \ge R_{\text{target}} \text{ (例: } +10\% \text{，鎖住當月果實)} \\ 1.0 \quad (\text{標準正常運行}) & \text{其他情況} \end{cases}
   $$
3. **綜合動態分數凱利與離散倉位映射（Composite Ratcheted Kelly Sizing）**：
   $$
   f^*_t = \hat{p}_{\text{cal}, t} - \frac{1 - \hat{p}_{\text{cal}, t}}{b_{\text{net}}}, \quad f_{\text{final}, t} = \max\left(0, \; \min\left(c \cdot \eta_{\text{vol}, t} \cdot \eta_{\text{month}, t} \cdot f^*_t, \; \text{MaxCap}\right)\right)
   $$

    結合階梯式棘輪基準本金 $W_{\text{base}, t}$ 映射至離散本金級距 $\mathcal{G} = \{0, 100\text{u}, 200\text{u}, 500\text{u}, W_{\text{base}, t}\}$：
    
   $$
   \text{Margin}_t = \max\left\{ g \in \mathcal{G} \;\middle\vert\; g \le W_{\text{base}, t} \cdot f_{\text{final}, t} \right\}
   $$

---

## 6. 實盤持倉狀態機、API對帳與超參數高原檢驗

### 6.1 含交易所對帳與孤兒隔離之實盤狀態機 (Reconciled Position State Machine)

```
 [ 每 5m K線啟動: 執行 REST API 對帳 (Reconciliation) 強制同步實際倉位与掛單 ]
                                      │
          [ 訊號觸發 E[R] > δ ]        ▼        [ Maker 限價單成交 ]
  ( IDLE ) ──────────────────► ( PENDING_ORDER ) ──────────────────► ( POSITION_OPEN )
     ▲                                │                                      │
     │      [ 超過 K 根未成交取消 ]     │         [ 綁定舊模型 Contract_k 孤兒隔離 ]
     └────────────────────────────────┘                                      ▼
     ▲                                                                 ( COOLDOWN )
     └────────────────────────── [ 冷卻期結束 & 檢查棘輪結算 ] ───────────────┘
```

### 6.2 超參數高原穩定度檢驗 (Parameter Plateau Objective)

$$
\mu_{\text{SR}}(\theta) = \frac{1}{\vert\mathcal{N}(\theta)\vert} \sum_{\theta' \in \mathcal{N}(\theta)} \text{SR}_{\text{OOS}}(\theta'), \quad \sigma_{\text{SR}}(\theta) = \sqrt{ \frac{1}{\vert\mathcal{N}(\theta)\vert} \sum_{\theta' \in \mathcal{N}(\theta)} \left( \text{SR}_{\text{OOS}}(\theta') - \mu_{\text{SR}}(\theta) \right)^2 }
$$

$$
J(\theta) = \mu_{\text{SR}}(\theta) - \lambda \cdot \sigma_{\text{SR}}(\theta) \quad (\text{建議懲罰係數 } \lambda = 1.0)
$$

---

## 7. 機構級收益驗證四大關卡 (Institutional Profit Validation Framework)

1. **第一關：降維夏普比率檢驗 (Deflated Sharpe Ratio, DSR)**：
   $$
   \text{SR}_0 = \sigma_{\{\text{SR}\}} \left[ (1 - \gamma)\Phi^{-1}\left(1 - \frac{1}{K_{\text{trials}}}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{K_{\text{trials}} \cdot e}\right) \right]
   $$

    
   $$
   \text{DSR} = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}_0)\sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}} \right) \ge 0.95
   $$
2. **第二關：蒙地卡羅路徑重抽樣壓力測試 (Monte Carlo Trade Reshuffling,** $B=10,000$**)**：
   - 第 95 百分位極端最大回撤：$\text{Quantile}_{0.95}(\text{MDD}) \le 20\%$；破產機率（跌破 $-30\%$）$< 0.5\%$。
3. **第三關：交易摩擦敏感度壓力測試 (Friction Sensitivity Stress Test)**：
   - 在 $1.5\times$ 雙邊摩擦成本下，年化夏普比率仍須 $> 1.0$ 且淨利留存率 $\ge 50\%$。
4. **第四關：月度滾動體感與利潤集中度指標 (Monthly Consistency &amp; Concentration)**：
   - 勝率月份佔比 $\in [65\%, 85\%]$、最長水下恢復期 $\text{Max TUW} \le 30\text{ 天}$、扣除 Top-5 獲利交易後之淨利留存率 $\ge 50\%$。

---

## 8. 完整 Python 實作藍圖 (Production-Grade Rolling Closed-Loop Code Blueprint v4.0)

以下提供完整可執行的模組化程式碼，包含跨週期無洩漏對齊、Meta-Labeling、樣本唯一性權重、**混合抽樣抗遺忘視窗**、**衛冕者-挑戰者無縫交接與孤兒持倉隔離控制器**、**非對稱階梯式棘輪複利與安全金庫**、強平保護與月度鎖利平滑引擎，以及**四大收益驗證關卡**：

```python
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression


# =====================================================================
# 1. 跨週期無洩漏對齊與特徵工程管線 (Multi-TF Feature Pipeline)
# =====================================================================
class MultiTimeframeFeaturePipeline:
    def __init__(self, lag_n: int = 5):
        self.lag_n = lag_n

    def _compute_single_tf_features(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """計算單一週期的無量綱 K 線微結構與定常態技術指標"""
        feats = pd.DataFrame(index=df.index)
        eps = 1e-8

        # A. K線微結構幾何 (Bar Geometry)
        feats[f'{prefix}_body'] = (df['close'] - df['open']) / (df['open'] + eps)
        bar_range = df['high'] - df['low'] + eps
        feats[f'{prefix}_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / bar_range
        feats[f'{prefix}_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / bar_range
        feats[f'{prefix}_norm_range'] = bar_range / (df['close'] + eps)

        for k in range(1, self.lag_n + 1):
            feats[f'{prefix}_log_ret_lag_{k}'] = np.log(df['close'] / df['close'].shift(k))

        vol_ma = df['volume'].rolling(window=20).mean() + eps
        feats[f'{prefix}_rvol'] = df['volume'] / vol_ma

        # B. 統計技術指標定常化 (Stationary Indicators)
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        ema_60 = df['close'].ewm(span=60, adjust=False).mean()
        feats[f'{prefix}_ema_bias_20'] = (df['close'] - ema_20) / (ema_20 + eps)
        feats[f'{prefix}_ema_bias_60'] = (df['close'] - ema_60) / (ema_60 + eps)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean() + eps
        rsi = 100.0 - (100.0 / (1.0 + gain / loss))
        feats[f'{prefix}_rsi_scaled'] = (rsi - 50.0) / 50.0

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        feats[f'{prefix}_norm_atr'] = atr_14 / (df['close'] + eps)

        mid = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std() + eps
        upper, lower = mid + 2.0 * std, mid - 2.0 * std
        feats[f'{prefix}_bb_pct'] = (df['close'] - lower) / (upper - lower + eps)
        feats[f'{prefix}_bb_bw'] = (upper - lower) / (mid + eps)

        # C. 合約專屬特徵
        if 'funding_rate' in df.columns:
            fr_mean = df['funding_rate'].rolling(30).mean()
            fr_std = df['funding_rate'].rolling(30).std() + eps
            feats[f'{prefix}_funding_z'] = (df['funding_rate'] - fr_mean) / fr_std
        if 'open_interest' in df.columns:
            feats[f'{prefix}_oi_delta'] = np.log((df['open_interest'] + eps) / (df['open_interest'].shift(1) + eps))

        return feats

    def build_multi_tf_features(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """以 5m 為主時間軸，自動重採樣 15m 與 30m 並嚴格對齊已收盤 Bar"""
        feats_5m = self._compute_single_tf_features(df_5m, prefix='tf5m')

        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        if 'funding_rate' in df_5m.columns:
            agg_dict['funding_rate'] = 'last'
        if 'open_interest' in df_5m.columns:
            agg_dict['open_interest'] = 'last'

        df_15m = df_5m.resample('15min', closed='left', label='left').agg(agg_dict).dropna()
        feats_15m = self._compute_single_tf_features(df_15m, prefix='tf15m')
        feats_15m.index = feats_15m.index + pd.Timedelta(minutes=15)

        df_30m = df_5m.resample('30min', closed='left', label='left').agg(agg_dict).dropna()
        feats_30m = self._compute_single_tf_features(df_30m, prefix='tf30m')
        feats_30m.index = feats_30m.index + pd.Timedelta(minutes=30)

        base_df = feats_5m.copy()
        base_df['close_time'] = base_df.index + pd.Timedelta(minutes=5)

        merged = pd.merge_asof(
            base_df.sort_values('close_time'),
            feats_15m.sort_index(),
            left_on='close_time', right_index=True, direction='backward'
        )
        merged = pd.merge_asof(
            merged.sort_values('close_time'),
            feats_30m.sort_index(),
            left_on='close_time', right_index=True, direction='backward'
        )
        merged.drop(columns=['close_time'], inplace=True)
        return merged.dropna()

    @staticmethod
    def prune_collinear_features(X: pd.DataFrame, threshold: float = 0.80) -> List[str]:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return [c for c in X.columns if c not in to_drop]


# =====================================================================
# 2. 兩階段元標籤與樣本唯一性權重引擎 (Meta-Labeling & Sample Weights)
# =====================================================================
class MetaLabelingEngine:
    def __init__(self, x: float = 0.03, y: float = 2.0, M: int = 15):
        self.x = x
        self.y = y
        self.M = M

    def generate_primary_signals(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=36, adjust=False).mean()
        mom = df['close'].pct_change(5)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[(ema_fast > ema_slow) & (mom > 0.002)] = 1
        signals[(ema_fast < ema_slow) & (mom < -0.002)] = -1
        return signals

    def apply_triple_barrier_and_weights(
        self, df: pd.DataFrame, primary_signals: pd.Series
    ) -> pd.DataFrame:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        sigs = primary_signals.values
        n = len(df)
        sl_pct = self.x / self.y

        meta_labels = np.full(n, np.nan)
        t_end_idx = np.full(n, -1, dtype=int)
        realized_ret = np.zeros(n)

        for i in range(n - self.M):
            d = sigs[i]
            if d == 0:
                continue

            entry_p = closes[i]
            tp_price = entry_p * (1.0 + d * self.x)
            sl_price = entry_p * (1.0 - d * sl_pct)

            outcome = 0
            end_pos = i + self.M
            ret = (closes[end_pos] - entry_p) / entry_p * d

            for offset in range(1, self.M + 1):
                curr_idx = i + offset
                h, l = highs[curr_idx], lows[curr_idx]

                if d == 1:
                    hit_tp, hit_sl = (h >= tp_price), (l <= sl_price)
                else:
                    hit_tp, hit_sl = (l <= tp_price), (h >= sl_price)

                if hit_tp and hit_sl:
                    outcome = 0
                    end_pos = curr_idx
                    ret = -sl_pct
                    break
                elif hit_tp:
                    outcome = 1
                    end_pos = curr_idx
                    ret = self.x
                    break
                elif hit_sl:
                    outcome = 0
                    end_pos = curr_idx
                    ret = -sl_pct
                    break

            meta_labels[i] = outcome
            t_end_idx[i] = end_pos
            realized_ret[i] = ret

        concurrency = np.zeros(n, dtype=float)
        valid_indices = np.where(~np.isnan(meta_labels))[0]
        for idx in valid_indices:
            concurrency[idx : t_end_idx[idx] + 1] += 1.0

        uniqueness = np.full(n, np.nan)
        raw_weights = np.full(n, np.nan)
        for idx in valid_indices:
            span_c = concurrency[idx : t_end_idx[idx] + 1]
            u_i = np.mean(1.0 / np.maximum(span_c, 1.0))
            uniqueness[idx] = u_i
            raw_weights[idx] = u_i * (np.abs(realized_ret[idx]) + 1e-5)

        mean_w = np.nanmean(raw_weights)
        norm_weights = raw_weights / (mean_w if mean_w > 0 else 1.0)

        res = pd.DataFrame({
            'primary_dir': sigs,
            'meta_label': meta_labels,
            't_end_pos': t_end_idx,
            'realized_ret': realized_ret,
            'uniqueness': uniqueness,
            'sample_weight': norm_weights
        }, index=df.index)
        return res.dropna()


# =====================================================================
# 3. 非對稱階梯式棘輪複利與安全金庫 (Asymmetric Compounding Ratchet)
# =====================================================================
class AsymmetricCompoundingRatchet:
    """
    管理活躍交易資金池 (Active Pool) 與不可動用安全金庫 (Cold Reserve Vault)
    實現「賺錢階梯式抽水鎖定、賠錢即時縮表防守」的非對稱棘輪效應
    """
    def __init__(self, initial_capital: float = 1000.0, reinvest_ratio: float = 0.60):
        self.w0 = initial_capital
        self.alpha = reinvest_ratio
        self.active_pool = initial_capital
        self.reserve_vault = 0.0
        self.ratchet_hwm = initial_capital
        self.period_start_active = initial_capital

    def update_intraday_pnl(self, net_pnl_u: float) -> None:
        """每筆交易平倉時更新活躍資金池"""
        self.active_pool = max(0.0, self.active_pool + net_pnl_u)

    def settle_rolling_period(self, is_flat: bool = True) -> Dict[str, float]:
        """
        在每個滾動週期 (如每 7/14 天) 且空倉時執行棘輪結算：
        將超越歷史高水位之新增利潤的 (1 - alpha) 抽水轉入安全金庫
        """
        if not is_flat:
            return self.get_status()

        total_realized = self.active_pool + self.reserve_vault
        if total_realized > self.ratchet_hwm:
            new_profit = total_realized - self.ratchet_hwm
            skimmed_to_vault = (1.0 - self.alpha) * new_profit
            self.reserve_vault += skimmed_to_vault
            self.active_pool -= skimmed_to_vault
            self.ratchet_hwm = total_realized

        self.period_start_active = self.active_pool
        return self.get_status()

    def get_ratcheted_base_capital(self) -> float:
        """計算當下用於代入凱利公式的階梯式有效基準本金 W_base"""
        intraday_dd = max(0.0, self.period_start_active - self.active_pool)
        locked_active_growth = self.alpha * max(0.0, self.ratchet_hwm - self.w0)
        w_base = self.w0 + locked_active_growth - intraday_dd
        return float(max(0.0, min(w_base, self.active_pool)))

    def get_status(self) -> Dict[str, float]:
        return {
            'total_equity': self.active_pool + self.reserve_vault,
            'active_pool': self.active_pool,
            'reserve_vault': self.reserve_vault,
            'ratchet_hwm': self.ratchet_hwm,
            'kelly_base_capital': self.get_ratcheted_base_capital()
        }


# =====================================================================
# 4. 滾動抗遺忘視窗、衛冕者/挑戰者交接與孤兒持倉隔離引擎
# =====================================================================
@dataclass
class PositionContract:
    """封裝開倉當下的模型參數契約，確保新舊模型交接時孤兒持倉隔離"""
    position_id: str
    model_id: str
    direction: int
    entry_price: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    open_bar_idx: int
    max_hold_bars: int


class RollingClosedLoopController:
    """
    管理混合抽樣抗遺忘視窗、Champion-Challenger 自動評估晉升與孤兒持倉隔離
    """
    def __init__(self, stress_ratio: float = 0.30, dsr_threshold: float = 0.90):
        self.stress_ratio = stress_ratio
        self.dsr_threshold = dsr_threshold
        self.champion_model: Optional[lgb.LGBMClassifier] = None
        self.champion_calibrator: Optional[IsotonicRegression] = None
        self.champion_id: str = "INIT_CHAMPION_v0"
        self.champion_params: Dict[str, float] = {'x': 0.025, 'y': 2.0, 'M': 15}
        self.kelly_penalty_scalar: float = 1.0
        self.open_contracts: Dict[str, PositionContract] = {}

    def build_hybrid_anti_amnesia_dataset(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        w_full: np.ndarray,
        recent_window_size: int = 2500,
        atr_col: str = 'tf5m_norm_atr'
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        合成 70% 近期連續樣本 + 30% 歷史極端高波動壓力記憶樣本 (防止 Regime Amnesia)
        """
        n_total = len(X_full)
        if n_total <= recent_window_size:
            return X_full, y_full, w_full

        # 1. 近期連續視窗 (Recent Window)
        X_recent = X_full.iloc[-recent_window_size:]
        y_recent = y_full.iloc[-recent_window_size:]
        w_recent = w_full[-recent_window_size:]

        # 2. 歷史高波動極端壓力庫 (Historical Stress Pool)
        X_hist = X_full.iloc[:-recent_window_size]
        y_hist = y_full.iloc[:-recent_window_size]
        w_hist = w_full[:-recent_window_size]

        if atr_col in X_hist.columns and len(X_hist) > 50:
            stress_cutoff = np.percentile(X_hist[atr_col].values, 85)
            stress_mask = (X_hist[atr_col] >= stress_cutoff).values
            stress_indices = np.where(stress_mask)[0]
        else:
            stress_indices = np.arange(len(X_hist))

        n_stress_target = int(recent_window_size * (self.stress_ratio / (1.0 - self.stress_ratio)))
        rng = np.random.default_rng(42)
        if len(stress_indices) > 0:
            chosen_idx = rng.choice(
                stress_indices,
                size=min(n_stress_target, len(stress_indices)),
                replace=False
            )
            X_stress = X_hist.iloc[chosen_idx]
            y_stress = y_hist.iloc[chosen_idx]
            w_stress = w_hist[chosen_idx]

            X_hybrid = pd.concat([X_stress, X_recent], axis=0)
            y_hybrid = pd.concat([y_stress, y_recent], axis=0)
            w_hybrid = np.concatenate([w_stress, w_recent])
            return X_hybrid, y_hybrid, w_hybrid

        return X_recent, y_recent, w_recent

    def train_and_evaluate_challenger(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        w_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_realized_rets: np.ndarray,
        candidate_id: str,
        candidate_params: Dict[str, float]
    ) -> Dict[str, object]:
        """
        訓練挑戰者模型 (Challenger) 並與現任衛冕者 (Champion) 進行樣本外對決與交接審核
        """
        split_cal = int(len(X_train) * 0.85)
        X_tr_sub, y_tr_sub, w_tr_sub = X_train.iloc[:split_cal], y_train.iloc[:split_cal], w_train[:split_cal]
        X_cal_sub, y_cal_sub = X_train.iloc[split_cal:], y_train.iloc[split_cal:]

        chal_clf = lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1
        )
        chal_clf.fit(X_tr_sub, y_tr_sub, sample_weight=w_tr_sub)

        chal_iso = IsotonicRegression(out_of_bounds='clip')
        chal_iso.fit(chal_clf.predict_proba(X_cal_sub)[:, 1], y_cal_sub)

        # 評估挑戰者在 OOS 驗證集上的期望表現
        chal_probs = chal_iso.transform(chal_clf.predict_proba(X_val)[:, 1])
        chal_active_rets = val_realized_rets[chal_probs > 0.55] - 0.0010
        chal_ev = float(np.mean(chal_active_rets)) if len(chal_active_rets) > 5 else -0.01

        dsr_info = ProfitValidationSuite.compute_deflated_sharpe_ratio(
            chal_active_rets if len(chal_active_rets) > 10 else val_realized_rets - 0.0010,
            num_trials=30, trials_sharpe_std=0.20
        )
        chal_dsr = dsr_info['dsr']

        # 若尚無衛冕者，直接上線
        if self.champion_model is None:
            self.champion_model = chal_clf
            self.champion_calibrator = chal_iso
            self.champion_id = candidate_id
            self.champion_params = candidate_params
            self.kelly_penalty_scalar = 1.0
            return {'status': 'INITIAL_CHAMPION_DEPLOYED', 'active_id': self.champion_id, 'chal_ev': chal_ev}

        # 計算現任衛冕者在相同驗證集上的表現
        champ_probs = self.champion_calibrator.transform(self.champion_model.predict_proba(X_val)[:, 1])
        champ_active_rets = val_realized_rets[champ_probs > 0.55] - 0.0010
        champ_ev = float(np.mean(champ_active_rets)) if len(champ_active_rets) > 5 else -0.01

        # 晉升條件：挑戰者期望收益 > 0 且優於衛冕者 (或 DSR 顯著達標)
        if chal_ev > 0.0 and (chal_ev >= champ_ev or chal_dsr >= self.dsr_threshold):
            prev_id = self.champion_id
            self.champion_model = chal_clf
            self.champion_calibrator = chal_iso
            self.champion_id = candidate_id
            self.champion_params = candidate_params
            self.kelly_penalty_scalar = 1.0
            return {
                'status': 'CHALLENGER_PROMOTED',
                'prev_id': prev_id,
                'active_id': self.champion_id,
                'chal_ev': chal_ev,
                'champ_ev': champ_ev,
                'orphan_positions_locked': len(self.open_contracts)
            }
        else:
            # 挑戰者未達標：保留原衛冕者，但啟動 0.75x 防禦性降檔
            self.kelly_penalty_scalar = 0.75
            return {
                'status': 'CHAMPION_RETAINED_WITH_THROTTLE',
                'active_id': self.champion_id,
                'chal_ev': chal_ev,
                'champ_ev': champ_ev,
                'kelly_penalty_scalar': self.kelly_penalty_scalar
            }

    def register_new_position(
        self, pos_id: str, direction: int, entry_price: float, margin: float, leverage: float, bar_idx: int
    ) -> PositionContract:
        """開立新倉時，將當前衛冕者的 (x, y, M) 封裝寫入獨立契約"""
        x = self.champion_params['x']
        y = self.champion_params['y']
        M = int(self.champion_params['M'])
        sl_pct = x / y

        tp_p = entry_price * (1.0 + direction * x)
        sl_p = entry_price * (1.0 - direction * sl_pct)

        contract = PositionContract(
            position_id=pos_id,
            model_id=self.champion_id,
            direction=direction,
            entry_price=entry_price,
            margin=margin,
            leverage=leverage,
            tp_price=tp_p,
            sl_price=sl_p,
            open_bar_idx=bar_idx,
            max_hold_bars=M
        )
        self.open_contracts[pos_id] = contract
        return contract


# =====================================================================
# 5. 合約強平防禦、分數凱利與月度平滑引擎 (Risk & Monthly Smoothing Engine)
# =====================================================================
class PerpetualRiskAndSmoothingEngine:
    def __init__(
        self,
        kelly_c: float = 0.25,
        max_cap: float = 0.20,
        mmr: float = 0.004,
        liq_buffer: float = 0.005,
        target_atr_norm: float = 0.004,
        monthly_profit_target: float = 0.10,
        monthly_soft_dd: float = 0.03,
        monthly_hard_dd: float = 0.06,
        discrete_buckets: Optional[List[float]] = None
    ):
        self.c = kelly_c
        self.max_cap = max_cap
        self.mmr = mmr
        self.liq_buffer = liq_buffer
        self.target_atr_norm = target_atr_norm
        self.monthly_profit_target = monthly_profit_target
        self.monthly_soft_dd = monthly_soft_dd
        self.monthly_hard_dd = monthly_hard_dd
        self.buckets = sorted(discrete_buckets) if discrete_buckets else None

    def compute_monthly_and_vol_scalars(
        self,
        current_equity: float,
        month_start_equity: float,
        month_hwm_equity: float,
        current_norm_atr: float
    ) -> Tuple[float, float, str]:
        eta_vol = float(np.clip(self.target_atr_norm / (current_norm_atr + 1e-8), 0.5, 1.5))
        mtd_return = (current_equity - month_start_equity) / (month_start_equity + 1e-8)
        mtd_dd = (month_hwm_equity - current_equity) / (month_hwm_equity + 1e-8)

        if mtd_return <= -self.monthly_hard_dd or mtd_dd >= (self.monthly_hard_dd + 0.02):
            eta_month = 0.0
            regime_mode = "MONTHLY_HALT_CIRCUIT_BREAKER"
        elif mtd_return <= -self.monthly_soft_dd:
            eta_month = 0.5
            regime_mode = "MONTHLY_DEFENSIVE_THROTTLE"
        elif mtd_return >= self.monthly_profit_target:
            eta_month = 0.5
            regime_mode = "MONTHLY_PROFIT_LOCK_SKIMMING"
        else:
            eta_month = 1.0
            regime_mode = "MONTHLY_NORMAL"

        return eta_vol, eta_month, regime_mode

    def compute_safe_sizing(
        self,
        prob_cal: float,
        direction: int,
        x: float,
        y: float,
        target_leverage: float,
        ratcheted_base_capital: float,
        total_equity: float,
        month_start_equity: float,
        month_hwm_equity: float,
        rolling_penalty_scalar: float = 1.0,
        current_norm_atr: float = 0.004,
        funding_rate: float = 0.0001,
        expected_funding_counts: int = 1,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage: float = 0.0003
    ) -> Dict[str, object]:
        sl_pct = x / y
        funding_cost = direction * funding_rate * expected_funding_counts

        cost_win = maker_fee + maker_fee + funding_cost
        cost_loss = maker_fee + taker_fee + slippage + funding_cost

        net_win = x - cost_win
        net_loss = sl_pct + cost_loss

        eta_vol, eta_month, month_mode = self.compute_monthly_and_vol_scalars(
            total_equity, month_start_equity, month_hwm_equity, current_norm_atr
        )

        if net_win <= 0 or net_loss <= 0 or eta_month == 0.0:
            return {
                'margin': 0.0, 'leverage': 0.0, 'exposure': 0.0,
                'ev': -1.0, 'month_mode': month_mode
            }

        ev = prob_cal * net_win - (1.0 - prob_cal) * net_loss
        if ev <= 0.0015:
            return {
                'margin': 0.0, 'leverage': 0.0, 'exposure': 0.0,
                'ev': ev, 'month_mode': month_mode
            }

        max_safe_lev = np.floor(1.0 / (net_loss + self.mmr + self.liq_buffer))
        actual_leverage = float(min(target_leverage, max(1.0, max_safe_lev)))

        b_net = net_win / net_loss
        f_star = prob_cal - (1.0 - prob_cal) / b_net
        if f_star <= 0.0:
            return {
                'margin': 0.0, 'leverage': actual_leverage, 'exposure': 0.0,
                'ev': ev, 'month_mode': month_mode
            }

        # 整合分數凱利、波動率縮放、月度鎖利與滾動交接懲罰係數
        f_final = min(f_star * self.c * eta_vol * eta_month * rolling_penalty_scalar, self.max_cap)
        raw_margin = ratcheted_base_capital * f_final

        if self.buckets:
            valid_b = [b for b in self.buckets if b <= raw_margin]
            final_margin = float(max(valid_b)) if valid_b else 0.0
        else:
            final_margin = raw_margin

        exposure = final_margin * actual_leverage
        return {
            'margin': final_margin,
            'leverage': actual_leverage,
            'exposure': exposure,
            'ev': ev,
            'eta_vol': eta_vol,
            'eta_month': eta_month,
            'month_mode': month_mode
        }


# =====================================================================
# 6. 機構級收益驗證四大關卡套件 (Institutional Profit Validation Suite)
# =====================================================================
class ProfitValidationSuite:
    @staticmethod
    def compute_deflated_sharpe_ratio(
        returns: np.ndarray,
        num_trials: int = 100,
        trials_sharpe_std: float = 0.5
    ) -> Dict[str, float]:
        T = len(returns)
        if T < 10 or np.std(returns) == 0:
            return {'sr_est': 0.0, 'sr_null': 0.0, 'dsr': 0.0}

        sr_est = float(np.mean(returns) / np.std(returns, ddof=1))
        sk = float(skew(returns))
        kr = float(kurtosis(returns, fisher=False))

        euler_gamma = 0.5772156649
        K = max(2, num_trials)
        z1 = norm.ppf(1.0 - 1.0 / K)
        z2 = norm.ppf(1.0 - 1.0 / (K * np.e))
        sr_null = trials_sharpe_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

        denom_var = 1.0 - sk * sr_est + ((kr - 1.0) / 4.0) * (sr_est ** 2)
        denom = np.sqrt(max(denom_var, 1e-8))
        z_score = ((sr_est - sr_null) * np.sqrt(T - 1.0)) / denom
        dsr = float(norm.cdf(z_score))

        return {'sr_est': sr_est, 'sr_null': sr_null, 'dsr': dsr}

    @staticmethod
    def run_monte_carlo_reshuffling(
        trade_returns: np.ndarray,
        n_simulations: int = 10000,
        ruin_threshold: float = 0.30
    ) -> Dict[str, float]:
        n_trades = len(trade_returns)
        if n_trades == 0:
            return {'mdd_median': 0.0, 'mdd_95th': 0.0, 'risk_of_ruin': 0.0}

        rng = np.random.default_rng(42)
        sampled_indices = rng.integers(0, n_trades, size=(n_simulations, n_trades))
        sim_rets = trade_returns[sampled_indices]

        equity_curves = np.cumprod(1.0 + sim_rets, axis=1)
        peaks = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (peaks - equity_curves) / np.maximum(peaks, 1e-8)
        max_dds = np.max(drawdowns, axis=1)

        return {
            'mdd_median': float(np.median(max_dds)),
            'mdd_95th': float(np.percentile(max_dds, 95)),
            'risk_of_ruin': float(np.mean(max_dds >= ruin_threshold))
        }


# =====================================================================
# 7. 端到端 v4.0 滾動閉環、模型交接、棘輪複利與孤兒持倉隔離示範
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    bars = 10000
    dates = pd.date_range("2025-01-01", periods=bars, freq="5min")

    ret = np.random.normal(0.00004, 0.0028, bars)
    close_p = 1000.0 * np.exp(np.cumsum(ret))
    high_p = close_p * (1.0 + np.abs(np.random.normal(0, 0.0022, bars)))
    low_p = close_p * (1.0 - np.abs(np.random.normal(0, 0.0022, bars)))
    open_p = np.roll(close_p, 1)
    open_p[0] = 1000.0

    df_5m = pd.DataFrame({
        'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p,
        'volume': np.random.lognormal(10, 0.6, bars),
        'funding_rate': np.random.normal(0.0001, 0.00005, bars),
        'open_interest': np.cumsum(np.random.normal(100, 500, bars)) + 500000
    }, index=dates)

    # 1. 建立跨週期特徵與元標籤
    pipe = MultiTimeframeFeaturePipeline(lag_n=3)
    X_all = pipe.build_multi_tf_features(df_5m)
    kept_cols = pipe.prune_collinear_features(X_all, threshold=0.80)
    X_pruned = X_all[kept_cols]

    meta_engine = MetaLabelingEngine(x=0.025, y=2.0, M=15)
    prim_signals = meta_engine.generate_primary_signals(df_5m)
    meta_df = meta_engine.apply_triple_barrier_and_weights(df_5m, prim_signals)

    common_idx = X_pruned.index.intersection(meta_df.index)
    X_model = X_pruned.loc[common_idx].copy()
    X_model['primary_dir'] = meta_df.loc[common_idx, 'primary_dir']
    y_model = meta_df.loc[common_idx, 'meta_label'].astype(int)
    w_model = meta_df.loc[common_idx, 'sample_weight'].values
    r_model = meta_df.loc[common_idx, 'realized_ret'].values

    # 2. 初始化滾動閉環控制器與非對稱棘輪資金池
    rolling_ctrl = RollingClosedLoopController(stress_ratio=0.30, dsr_threshold=0.85)
    ratchet_vault = AsymmetricCompoundingRatchet(initial_capital=1000.0, reinvest_ratio=0.60)
    risk_engine = PerpetualRiskAndSmoothingEngine(
        kelly_c=0.35, max_cap=0.50, discrete_buckets=[100.0, 200.0, 500.0, 1000.0]
    )

    # 第一期滾動部署 (Cycle 1: 部署初始衛冕者 Champion_v1)
    split_1, split_2 = int(len(X_model) * 0.55), int(len(X_model) * 0.75)
    X_hyb_1, y_hyb_1, w_hyb_1 = rolling_ctrl.build_hybrid_anti_amnesia_dataset(
        X_model.iloc[:split_1], y_model.iloc[:split_1], w_model[:split_1], recent_window_size=1200
    )
    res_cycle1 = rolling_ctrl.train_and_evaluate_challenger(
        X_hyb_1, y_hyb_1, w_hyb_1,
        X_model.iloc[split_1:split_2], y_model.iloc[split_1:split_2], r_model[split_1:split_2],
        candidate_id="MODEL_CYCLE_1", candidate_params={'x': 0.025, 'y': 2.0, 'M': 15}
    )

    # 模擬第一期開立一筆倉位 (將在交接時成為孤兒持倉)
    orphan_contract = rolling_ctrl.register_new_position(
        pos_id="POS_001", direction=1, entry_price=1050.0, margin=200.0, leverage=3.0, bar_idx=split_2 - 2
    )

    # 模擬第一期實現獲利 +300u，並在空倉點執行棘輪抽水結算
    ratchet_vault.update_intraday_pnl(+300.0)
    ratchet_status = ratchet_vault.settle_rolling_period(is_flat=True)

    # 第二期滾動重訓與衛冕者/挑戰者對決 (Cycle 2: 推進視窗訓練 Challenger_v2)
    split_3 = int(len(X_model) * 0.90)
    X_hyb_2, y_hyb_2, w_hyb_2 = rolling_ctrl.build_hybrid_anti_amnesia_dataset(
        X_model.iloc[:split_2], y_model.iloc[:split_2], w_model[:split_2], recent_window_size=1200
    )
    res_cycle2 = rolling_ctrl.train_and_evaluate_challenger(
        X_hyb_2, y_hyb_2, w_hyb_2,
        X_model.iloc[split_2:split_3], y_model.iloc[split_2:split_3], r_model[split_2:split_3],
        candidate_id="MODEL_CYCLE_2", candidate_params={'x': 0.035, 'y': 2.5, 'M': 20}
    )

    # 3. 結合棘輪基準本金與最新活躍模型進行實盤推論
    latest_X = X_model.iloc[[-1]]
    latest_dir = int(latest_X['primary_dir'].iloc[0])
    raw_prob = rolling_ctrl.champion_model.predict_proba(latest_X)[:, 1][0]
    cal_prob = float(rolling_ctrl.champion_calibrator.transform([raw_prob])[0])

    sizing_res = risk_engine.compute_safe_sizing(
        prob_cal=cal_prob,
        direction=latest_dir,
        x=rolling_ctrl.champion_params['x'],
        y=rolling_ctrl.champion_params['y'],
        target_leverage=5.0,
        ratcheted_base_capital=ratchet_status['kelly_base_capital'],
        total_equity=ratchet_status['total_equity'],
        month_start_equity=1000.0,
        month_hwm_equity=1300.0,
        rolling_penalty_scalar=rolling_ctrl.kelly_penalty_scalar,
        current_norm_atr=float(latest_X['tf5m_norm_atr'].iloc[0])
    )

    print("=== 系統 v4.0 全自動滾動閉環、模型交接與階梯式棘輪報告 ===")
    print(f"[Cycle 1 部署] 狀態: {res_cycle1['status']} | 線上模型: {res_cycle1['active_id']}")
    print(f"[Cycle 2 交接] 狀態: {res_cycle2['status']} | 當前衛冕者: {res_cycle2['active_id']}")
    print(f"[孤兒持倉隔離] 既有持倉 {orphan_contract.position_id} 綁定模型: {orphan_contract.model_id} | 鎖定止盈價: {orphan_contract.tp_price:.2f} (不受新模型參數影響)")
    print(f"[非對稱棘輪金庫] 總資產: {ratchet_status['total_equity']:.0f} U | 活躍交易池: {ratchet_status['active_pool']:.0f} U | 安全冷金庫鎖定: {ratchet_status['reserve_vault']:.0f} U | 凱利基準本金 W_base: {ratchet_status['kelly_base_capital']:.0f} U")
    print(f"[最新推論核准] 校準勝率: {cal_prob*100:.2f}% | 當月模式: {sizing_res['month_mode']} | 核准保證金: {sizing_res['margin']:.0f} U | 安全槓桿: {sizing_res['leverage']:.1f}x")
```

---

## 9. 滾動閉環、實盤風控與收益驗證准入矩陣 (v4.0 Master Governance Matrix)


| 類別         | 監控 / 驗證維度                               | 觸發條件 / 准入閾值                                                                                                        | 自動化熔斷與處置措施                                                                  |
| :---------- | :--------------------------------------- | :------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------- |
| **滾動交接治理** | **衛冕者-挑戰者對決 (Champion-Challenger)**     | $\mathbb{E}[R_{\text{OOS}}(\mathcal{M}_{\text{chal}})] \le \mathbb{E}[R_{\text{OOS}}(\mathcal{M}_{\text{champ}})]$ | **拒絕盲目替換模型**，保留原衛冕者並將凱利注碼乘以 $0.75$ 防禦降檔                                     |
| **滾動交接治理** | **孤兒持倉隔離 (Orphan Position Isolation)**  | 新模型晉升時帳戶存在未平倉部位 $\text{Pos}_{\text{old}}$                                                                          | 既有部位嚴格鎖定開倉時封裝的 $\text{Contract}_k(B_{\text{TP}}, B_{\text{SL}}, M)$，禁止新模型覆寫 |
| **滾動交接治理** | **報復性過擬合熔斷 (Revenge Overfitting)**      | Loop 3 所有參數高原之 $\max_\theta \mu_{\text{SR}}(\theta) < 0.5$                                                         | 判定市場處於無趨勢噪音期，輸出 **`PASS_REGIME`** 全系統休眠空倉                                   |
| **資金複利治理** | **非對稱棘輪抽水 (Asymmetric Profit Ratchet)** | 滾動結算淨值創歷史新高 $\Delta \Pi_k > 0$                                                                                     | 自動將 $(1 - \alpha)$（如 $40\%$）新增利潤轉入不可動用冷金庫 $V_{\text{vault}}$，永久鎖死下限         |
| **實盤月度平滑** | **月度達標鎖利 (Monthly Profit Skimming)**    | 當月累計收益 $R_{\text{MTD}} \ge +10\%$                                                                                  | 自動將當月凱利縮放係數減半（$\eta_{\text{month}} = 0.5$），進入「保果實模式」                        |
| **實盤月度平滑** | **月度硬底線熔斷 (Monthly Circuit Breaker)**   | 當月累計虧損 $\le -6\%$ 或自月高點回撤 $\ge 8\%$                                                                                | 當月強制停止開新倉（$\eta_{\text{month}} = 0$），鎖定帳戶底線至下月初重置                           |
| **實盤微觀風控** | **強平安全邊界 (Liquidation Guard)**          | $\frac{x}{y} + \text{Cost} + \text{Buffer} > \frac{1}{z} - \text{MMR}$                                             | 自動將該筆訂單槓桿下調至 $z_{\text{safe}}$，杜絕停損前遭強平                                     |
| **離線收益驗證** | **四大驗證關卡 (DSR / MC / 摩擦 / 月度)**         | $\text{DSR} < 0.95$ 或 $\text{MC MDD}_{95\%} > 20\%$ 或 $\text{TUW} > 30\text{天}$                                    | 判定模型未通過機構級壓力測試，**禁止晉升上線**                                                   |

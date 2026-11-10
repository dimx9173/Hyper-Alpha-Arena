# ICT / SMC 指標模組

`backend/services/ict_smc_indicators.py` 實作 Smart Money Concepts 的價格結構分析。

## 為什麼需要

GA 目前搜尋的參數只涵蓋一種策略結構：EMA 趨勢 + Donchian 突破 + RSI 過濾 + 固定 TP/SL。
ICT/SMC 是不同的流派（價格結構分析），要納入搜尋需要先把主觀概念量化成可回測規則。

本模組是**下一輪 GA 擴充參數空間的基礎設施**，目前尚未接入 GA 基因。

## 實作的量化定義

ICT/SMC 許多概念在不同文獻中定義不一致。這個模組把每個定義寫死並附上理由，
避免「主觀判斷」被硬編成無法解釋的回測規則。

| 概念 | 量化定義 |
|---|---|
| **Swing Point** | `high[i]` 是 `[i-left, i+right]` 區間內唯一最高點（low 對稱） |
| **BOS** | 突破最近一個已確認且未破的 swing high（down 反之） |
| **CHoCH** | 與當前結構方向**相反**的突破 |
| **Order Block** | 造成突破的那根反向實體 K 線，`mitigated` 標記是否已被回補 |
| **Fair Value Gap** | 三根連續同向 K 線，中間根需有明顯實體（`min_body_ratio` 可調） |
| **Liquidity Sweep** | 跌破/掃過 swing 後收回，且影線 ≥ 實體 × `wick_ratio` |
| **Premium / Discount** | 價格在近 lookback 區間內的相對位置（≥0.5 premium） |
| **OTE** | CHoCH + Order Block + FVG + Sweep + 區間位置的加權 confluence（0~1） |

## 前視偏差保證

`swing` 需要右側 `swing_right` 根確認才能被「知道」。掃描在每個索引 `i`
**只使用 `confirm_index <= i` 的 swing**。

這個保證是被測試抓出來的。原始實作先把參考 swing 設成「整段資料最後一個
已確認 swing」再從早期索引開始掃描，導致 index 4 就用到了 index 8 之後才
確認的 swing — 同一根 K 線在截斷序列與完整序列中會被判成不同事件。

`test_no_lookahead_structure` 逐一根驗證這個性質，測試資料用 77 根 K 線，
每個時點都重算並比對最後一個事件。

## 參數預設值經實測校準

`swing_left` / `swing_right` 預設 **5**，不是文獻常見的 2。
BTC 500 根 1h K 線實測：

| left | right | swings | events | BOS | CHoCH |
|---|---|---|---|---|---|
| 2 | 2 | 140 | 58 | 36 | 22 | ← 噪音過多，CHoCH 失去意義 |
| 3 | 3 | 99 | 41 | 25 | 16 |
| 4 | 4 | 78 | 33 | 20 | 13 |
| **5** | **5** | **63** | **25** | **18** | **7** | ← 預設 |
| 6 | 6 | 53 | 21 | 14 | 7 |

`left`/`right` 應該視為可被 GA 搜尋的參數。1h 級別噪音大用 5；4h/1d 可降到 3。

## 策略端用法

```python
info = data.get_indicator(symbol, "ICT_OTE", "1h")
score = info.get("value")        # 0~1 confluence 分數

st = data.get_indicator(symbol, "ICT_STRUCTURE", "1h")
st.get("direction")              # "up" | "down" | None
st.get("last_event")             # 最後一次 BOS/CHoCH

# 可用後綴覆寫確認根數
data.get_indicator(symbol, "ICT_OTE_3", "1h")     # left=right=3
```

完整 payload（`"ICT"`）包含 swings / structure / order_blocks / fvgs /
sweeps / premium_discount / ote 七個子項，每個獨立 try/except，
單一偵測器失敗不讓整個指標失效。

## 訊號密度實測

8 個標的 × 30 個滾動時點 = 240 樣本：

| score | 樣本數 |
|---|---|
| ≥ 0.90 | 1 |
| ≥ 0.80 | 12 |
| ≥ 0.70 | 30（12.5%） |
| ≥ 0.55 | 97 |
| ≥ 0.45 | 222 |

OTE ≥ 0.7 在 12.5% 的時點出現 — 不是永不觸發，也不太頻繁。

## 測試

```bash
cd backend && uv run python test_ict_smc.py
```

17 個測試，涵蓋：偵測正確性、**前視偏差**、參數敏感度、退化資料、
邊界條件（0~6 根 K 線）、子項失敗隔離。

## 尚未完成

- **未接入 GA 基因**：`BOUNDS` 還沒有 ICT 相關參數，需要等當前 100 代跑完再改
- **未接入 `calculate_indicator()`**：`calculate_indicator()` 走 DB 單根計算，
  與 `calculate_indicators()` 的批次路徑不同，ICT 需要完整序列，接上時要注意
- **未做績效驗證**：模組本身正確，但「用 ICT 訊號下單能否賺錢」是另一個問題，
  必須經過 GA + 回測 + Annual Gate 才能回答

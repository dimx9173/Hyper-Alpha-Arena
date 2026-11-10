# Paper Trading — mainnet 行情 + 本地計算

## 解決的問題

2026-10-03 診斷出回測與實盤落差不是過擬合，而是**價格來源不一致**：

| 標的 | Binance | HL testnet | HL mainnet |
|---|---|---|---|
| ZEC | 1307.74 | **138.65** | 1305.25 |
| NIL | 0.0834 | 0.0835 | 0.0840 |
| HYPE | 88.41 | **60.16** | 88.39 |
| PUMP | 0.0058 | 0.0060 | 0.0058 |

- testnet 與 binance/mainnet 價格結構不同（ZEC 差 9.4 倍）
- 用 binance 資料回測、在 testnet 執行 → 回測績效無法預測實盤

**解法**：行情走 **Hyperliquid mainnet**（與 binance 一致，20/20 標的都在 0.3% 內），
資金與持倉 **完全本地計算**，不簽名、不廣播、不碰鏈上。

## 與 testnet paper 的差異

| | 舊（testnet） | 新（mainnet 行情 + 本地計算） |
|---|---|---|
| 行情來源 | testnet API | **mainnet API** |
| 價格 vs 回測 | 不一致（ZEC 差 9.4x） | **一致** |
| 下單 | 真的發鏈上交易 | **不發**，只更新本地帳本 |
| 資金 | testnet 測試幣 | 虛擬 10000 USDC |
| 滑價/手續費 | 鏈上真實 | **模型參數** |
| TP/SL 觸發 | 交易所托管 | **每次 check() 用現價判定** |

## 會計模型

**關鍵：保證金不是轉出資金。**

```
權益 = 錢包餘額 + 未實現損益
錢包餘額 = 初始資金 + 已實現損益 − 費用
可用餘額 = 錢包餘額 − 已佔用保證金
```

開倉時**不**從現金扣保證金。保證金是交易所要求的抵押品，扣掉會讓權益
憑空減少一整個 margin。這個錯誤會讓 equity identity 崩潰（實測偏差 3001 USDC）。

`test_equity_identity` 驗證這個恆等式精確成立（誤差 < 1e-6）。

## 成本模型（FeeModel）

預設取自 Hyperliquid mainnet 實際費率量級：

```python
maker_fee_bps = 1.5    # 0.015%
taker_fee_bps = 4.5    # 0.045%
slippage_bps  = 2.0    # 單邊 2bp
```

- 買入滑高、賣出滑低；買回滑低、賣出滑高
- TP/SL 觸發視為 maker（限價單），一般市價單為 taker
- 這些是**模型參數**，不是真實成交結果。績效數字是模型輸出。

## 風控護欄（與 program_execution_service 對齊）

| 護欄 | 上限 | 說明 |
|---|---|---|
| 總暴露 | ≤ 2.0x equity | 所有持倉 notional 合計 |
| 單幣 | ≤ 0.4x equity | 單一標的 |
| 同簇 | ≤ 0.6x equity | BTC/ETH/SOL/HYPE/BNB/AVAX/NEAR |
| 持倉數 | ≤ 3 | |
| 同向加倉冷卻 | 4 小時 | 避免訊號重複加倉 |

護欄是**事後檢查**：先算 size 再比對上限。所以 portion 的分母是
「開倉當下的 available_balance」，與真實執行路徑一致。

副作用：累積 portion 會因 available_balance 遞減而永遠碰不到 2x 上限。
要觸發總暴露護欄需要用高槓桿單筆大倉位（見 `test_exposure_cap_guard`）。

## TP/SL 判定

每次 `check_exits(marks)` 掃描所有持倉：

- 多頭：mark ≥ take_profit → 以 TP 價平倉
- 多頭：mark ≤ stop_loss → 以 SL 價平倉
- **同一根 K 線同時觸及時 TP 優先**（先檢查 TP）

TP 優先是保守假設。真實市場同一根 1h K 線內先觸及哪個價位未知，
這個選擇不會讓回測失真。

## 使用方式

```python
from services.paper_broker import PaperBroker, FeeModel

broker = PaperBroker(initial_capital=10000.0, fee_model=FeeModel())

# 開倉
r = broker.open_long("BTC", portion=0.22, mid_price=84770.5, leverage=1,
                     take_profit=94773.4, stop_loss=78836.6)
if not r["success"]:
    print("護欄擋下:", r["skipped"])   # 例如 single_coin_cap / cluster_cap

# 每次排程觸發時先檢查出場
marks = fetch_mainnet_prices(["BTC", "ETH", "SOL"])
exits = broker.check_exits(marks)

# 查帳務
state = broker.get_account_state(marks)
perf = broker.performance()

# 持久化
broker.save_to("/tmp/paper_state.json")
broker2 = PaperBroker.load_from("/tmp/paper_state.json")
```

## 測試

```bash
cd backend && uv run python test_paper_broker.py
```

17 個測試全通過，涵蓋：

- 會計恆等式精確成立
- TP/SL 觸發與優先順序
- 四種護欄（單幣/總暴露/同簇/持倉數）+ 同向冷卻
- 加倉加權平均成本
- 績效統計（勝率/PF/回撤）
- 持久化往返
- 邊界條件與零滑價精確盈虧

## 端到端驗證（mainnet 真實價格）

| 情境 | 權益 | 報酬 |
|---|---|---|
| 持倉不動 | 9997.64 | −0.024%（只有開倉費） |
| 全數止盈 (+12%) | 10615.54 | +6.16% |
| 全數止損 (−8%) | 9628.15 | −3.72% |

---

## 整合完成（2026-10-03）

### 三個整合任務全部完成

**1. `program_execution_service` 路由到 PaperBroker**

`_execute_binding` 新增 `environment == "paper"` 分支：

```python
elif environment == "paper":
    trading_client = PaperTradingClient(
        account_id=account.id, state_path=PAPER_STATE_PATH, ...)
    # 每次觸發先掃 TP/SL —— 鏈上没有掛單，本地必須自己判定
    closed = trading_client.check_exits()
```

關鍵細節：`data_env = "mainnet" if environment == "paper" else environment`。
若不這樣做，`get_price()` / `get_klines()` 會去 testnet 拿與回測不一致的價格。

**2. 排程整合**

`get_global_trading_mode()` 現在接受 `"paper"`。預設仍是 `"testnet"`
（安全性考量：不會意外動用真錢）。

每次排程觸發的流程：
```
觸發 → check_exits() 掃 TP/SL → 執行策略 → 護欄檢查 → 本地記帳
```

**3. API 端點**

| 端點 | 用途 |
|---|---|
| `GET /api/hyperliquid/paper/{account_id}` | 帳務 + 持倉 + 績效 |
| `POST /api/hyperliquid/paper/{account_id}/reset` | 清空帳戶 |
| `POST /api/hyperliquid/paper/{account_id}/check-exits` | 手動觸發 TP/SL 掃描 |
| `GET/POST /api/hyperliquid/trading-mode` | 切換 paper/testnet/mainnet |

### PaperTradingClient adapter

`services/paper_trading_client.py` 實作與真實 `TradingClient` 相同的介面，
讓執行服務不需要知道自己在 paper 模式。

關鍵：執行服務實際呼叫的是 `place_order_with_tpsl()` 不是 `place_order()`。
adapter 兩者都有。

TP/SL **不**做成獨立子單。鏈上那樣做有「主單成交但子單缺失」的風險
（PRP-P2-4 記錄過）。paper 模式直接記進持倉，由 `check_exits()` 用現價判定。

### Rate limit 修正

**問題**：策略掃 20 個標的，每個要 `fetch_ticker` + `candleSnapshot`，
40+ 次請求撞上 Hyperliquid 的 429。實測有 12 個標的讀不到價格，
策略決策嚴重劣化。

**解法**：`allMids` 端點一次回傳全部 3278 個合約的最新價。
加入 `_batch_price_cache`（TTL 60 秒），`get_last_price()` 先查快取。

實測效果：

| | 修正前 | 修正後 |
|---|---|---|
| 429 錯誤 | 12 筆 | **0** |
| 全部 ERROR | 多筆 | **0** |
| 讀到價格的標的 | ~8/20 | **20/20** |
| 執行時間 | 26s | 25s |

### 端到端驗證

用一個強制買進的測試策略走完整鏈路：

```
策略決策 buy BTC portion=0.20
  → 護欄檢查通過
  → 本地記帳：size=0.023584 entry=85226.57 margin=2010.01
  → 持倉追蹤：TP=89024.88 SL=82242.03
  → TP 觸發測試：exit=83937.65 net_pnl 正確計算（含手續費）
```

BTC 進場價 85226.57 來自 mainnet，與 binance 同源。

### 目前狀態

```
交易模式  paper
排程      binding 2（GA-100 gen78），每小時
帳戶      GA Best Paper，10000 USDC 虛擬資金
資料來源  Hyperliquid mainnet
測試      paper_broker 17/17、ict_smc 17/17
```

### 開發時修掉的 bug

| bug | 影響 |
|---|---|
| 保證金被重複扣減 | 權益低估 3001 USDC，equity identity 崩潰 |
| `available_balance()` 循環依賴 | portion 分母錯誤 |
| paper 模式被錢包檢查擋下 | 整條下單路徑走不通 |
| `get_last_price_from_hyperliquid("paper")` | 只認 testnet/mainnet，會拿不到價格 |
| adapter 缺 `place_order_with_tpsl` | 執行服務呼叫的是這個方法 |
| `cooldown_hours` 非建構參數 | 建構時 TypeError |
| 逐標的 API 呼叫 | 12/20 標的撞 rate limit |

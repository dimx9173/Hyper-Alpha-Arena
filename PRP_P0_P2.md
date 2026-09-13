# PRP: P0~P2 落地驗收總綱

目標: 把 kimi/agy 評審的 P0~P2 優化逐條落地, 每條獨立 commit + 可運行驗收。

## P0 (本週, 純工程+執行止血)
| ID | 條目 | 改動文件 | 驗收命令 |
|----|------|----------|----------|
| P0-1 | SL轉市價 + 回測SL滑點x3 | program_trader/models.py, services/hyperliquid_trading_client.py, backtest/execution_simulator.py, backtest/ab_validators.py | 單測SL路徑 + EMA定案重跑GateA(允許pnl下修但maxDD不惡化) |
| P0-2 | 指標增量快取 | backtest/historical_data_provider.py | 同策略前後回測結果一致 + 單輪耗時下降 |
| P0-3 | 變異去重+冒烟 | backtest/iterate.py | 連跑3輪無重複code_hash + 零交易輪被跳過 |
| P0-4 | 閘門早停 | backtest/iterate.py | rejected輪只跑GateA, log記停在哪一閘 |

## P1 (下週, 寫實化+穩健性)
| ID | 條目 | 改動文件 | 驗收命令 |
|----|------|----------|----------|
| P1-1 | funding扣費+過濾 | backtest/execution_simulator.py, backtest/historical_data_provider.py, best_code_ema_donch.py | shadow跑10輪看funding佔比, 再轉正 |
| P1-2 | TP/SL寫實 | backtest/execution_simulator.py | TP收盤確認+maker/taker分費, 排名重跑 |
| P1-3 | 觸發對齊+加倉冷卻 | backtest/ab_validators.py, services/program_execution_service.py | 4h策略trigger=3600, 同向加倉4h冷卻 |
| P1-4 | WF改制 | backtest/ab_validators.py | 盤整折相對buy_hold+熔斷-5% |

## P2 (隨後, 執行層)
| ID | 條目 | 驗收 |
|----|------|------|
| P2-1 | 流動性門控 | 剔LINK/UNI類無盤口幣+分檔滑點 |
| P2-2 | GTC TTL/撤單 | fallback單60-120s超時撤單 |
| P2-3 | 組合風控 | 總暴露/相關簇/日虧熔斷 |
| P2-4 | 監控閉環 | TP/SL對賬+心跳+連虧暫停+TG/Discord告警 |

## 驗收總則
- 每條獨立commit, 格式 `feat(prp-p0-1): ...` 
- 每條驗收輸出寫 /tmp/prp_<id>_accept.log
- 回歸: EMA定案 GateA 在 monthly10 門檻下不跌破 6/8

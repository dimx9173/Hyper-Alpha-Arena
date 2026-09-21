class Strategy:
    def should_trade(self, data):
        # v4 long-only — v3 relaxed: mom 0.3% trend 0.25% vol1.10x (放宽过紧过滤, 提升周转以达30%目标)
        # 仓位0.25x2, TP3.5%/SL1.5%, 保留ATR0.2%+阳线+RSI70, 预期提升25->40 trades/30d
        for symbol in ["BTC", "ETH", "HYPE", "SOL", "ZEC", "PUMP", "XRP", "LIT", "XMR", "PONS", "SKR", "kPEPE", "ENA", "UNI", "TRUMP", "NEAR", "FARTCOIN", "CASHCAT", "XPL", "TAO", "GRAM", "SUI", "DOGE", "LINK", "CRV", "AAVE", "ARB", "ADA", "BNB", "ZORA"]:
            price = data.get_price(symbol)
            if price == 0:
                continue
            klines = data.get_klines(symbol, "5m", 60)
            if len(klines) < 30:
                continue
            closes = [k.close for k in klines]
            volumes = [k.volume for k in klines]
            highs = [k.high for k in klines]
            lows = [k.low for k in klines]
            atr = sum(abs(highs[i]-lows[i]) for i in range(max(0,len(highs)-14), len(highs))) / 14 if highs else 0
            atr_ok = atr / price > 0.002 if price else False
            if not atr_ok:
                continue
            mom_20 = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 and closes[-20] else 0
            if mom_20 < 0.003:
                continue
            opens = [k.open for k in klines]
            if closes[-1] <= opens[-1]:
                continue
            ma_fast = sum(closes[-8:]) / 8
            ma_slow = sum(closes[-30:]) / 30
            trend_strength = abs(ma_fast - ma_slow) / ma_slow if ma_slow else 0
            if trend_strength < 0.0025:
                continue
            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0
            if not (volumes[-1] > vol_avg * 1.10 if vol_avg else False):
                continue
            recent_range = (max(highs[-5:]) - min(lows[-5:])) / price if price else 0
            if recent_range < 0.002:
                continue
            try:
                rsi_info = data.get_indicator(symbol, "RSI14", "5m")
                rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50
                if rsi is None:
                    rsi = 50
            except Exception:
                rsi = 50
            if rsi > 70:
                continue
            if ma_fast > ma_slow * 1.002:
                return Decision(operation="buy", symbol=symbol, reason="v3 long-only ATR+mom", target_portion_of_balance=0.34, leverage=2, max_price=price*1.005, take_profit_price=price*1.024, stop_loss_price=price*0.988)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

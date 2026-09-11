class Strategy:
    def should_trade(self, data):
        # constrained v2: max 3 coins each one position, no-add same coin, lev2, wide TP6%/SL2.5%, 1h trend
        try:
            open_pos = data.positions or {}
        except Exception:
            open_pos = {}
        try:
            n_open = len(open_pos)
        except Exception:
            n_open = 0
        if n_open >= 3:
            return Decision(operation="hold", symbol="BTC", reason="max-3-positions guard")
        skip = set()
        try:
            skip = set(open_pos.keys())
        except Exception:
            skip = set()
        for symbol in ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR", "SUI", "DOGE", "ARB"]:
            if symbol in skip:
                continue
            price = data.get_price(symbol)
            if price == 0:
                continue
            klines = data.get_klines(symbol, "1h", 80)
            if len(klines) < 50:
                continue
            closes = [k.close for k in klines]
            highs = [k.high for k in klines]
            lows = [k.low for k in klines]
            volumes = [k.volume for k in klines]
            ma_fast = sum(closes[-8:]) / 8
            ma_slow = sum(closes[-30:]) / 30
            if ma_slow <= 0:
                continue
            trend = (ma_fast - ma_slow) / ma_slow
            if trend < 0.006:
                continue
            mom_30 = (closes[-1] - closes[-30]) / closes[-30] if closes[-30] else 0
            if mom_30 < 0.012:
                continue
            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0
            if not (vol_avg and volumes[-1] > vol_avg * 1.25):
                continue
            atr = sum(abs(highs[i]-lows[i]) for i in range(len(highs)-14, len(highs))) / 14 if highs else 0
            if not (price and atr / price > 0.003):
                continue
            opens = [k.open for k in klines]
            if closes[-1] <= opens[-1]:
                continue
            if closes[-1] <= ma_fast:
                continue
            try:
                rsi_info = data.get_indicator(symbol, "RSI14", "1h")
                rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50
                if rsi is None:
                    rsi = 50
            except Exception:
                rsi = 50
            if rsi > 72 or rsi < 48:
                continue
            return Decision(operation="buy", symbol=symbol, reason="constrained 1h trend", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.06, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

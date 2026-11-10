class Strategy:
    def should_trade(self, data):
        # factor-C: MACD hist 4h trend + EMA50, lev2, TP8%/SL3%, max3 no-add
        try:
            open_pos = data.positions or {}
        except Exception:
            open_pos = {}
        try:
            n_open = len(open_pos)
        except Exception:
            n_open = 0
        if n_open >= 3:
            return Decision(operation="hold", symbol="BTC", reason="max-3 guard")
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
            klines = data.get_klines(symbol, "4h", 120)
            if len(klines) < 60:
                continue
            closes = [k.close for k in klines]
            try:
                m = data.get_indicator(symbol, "MACD", "4h")
                hist = m.get("histogram") if isinstance(m, dict) else None
            except Exception:
                hist = None
            if not hist:
                continue
            if hist <= 0:
                continue
            try:
                e = data.get_indicator(symbol, "EMA50", "4h")
                ema50 = e.get("value") if isinstance(e, dict) else None
            except Exception:
                ema50 = None
            if ema50 is None or closes[-1] <= ema50:
                continue
            mom_20 = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] else 0
            if mom_20 < 0.02:
                continue
            volumes = [k.volume for k in klines]
            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0
            if not (vol_avg and volumes[-1] > vol_avg * 1.2):
                continue
            try:
                rsi_info = data.get_indicator(symbol, "RSI14", "4h")
                rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50
                if rsi is None:
                    rsi = 50
            except Exception:
                rsi = 50
            if rsi > 75 or rsi < 50:
                continue
            return Decision(operation="buy", symbol=symbol, reason="macd 4h trend", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.08, stop_loss_price=price*0.97)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

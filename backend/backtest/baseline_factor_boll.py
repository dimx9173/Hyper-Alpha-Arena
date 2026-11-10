class Strategy:
    def should_trade(self, data):
        # factor-B: BOLL breakout 1h, lev2, TP6%/SL2.5%, max3 no-add
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
            klines = data.get_klines(symbol, "1h", 80)
            if len(klines) < 50:
                continue
            closes = [k.close for k in klines]
            try:
                b = data.get_indicator(symbol, "BOLL", "1h")
                upper = b.get("upper") if isinstance(b, dict) else None
                middle = b.get("middle") if isinstance(b, dict) else None
                lower = b.get("lower") if isinstance(b, dict) else None
            except Exception:
                upper = middle = lower = None
            if upper is None or lower is None:
                continue
            bw = (upper - lower) / price if price else 0
            if bw < 0.02 or bw > 0.15:
                continue
            if closes[-1] <= upper:
                continue
            if len(closes) >= 2 and closes[-2] > upper:
                continue
            volumes = [k.volume for k in klines]
            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0
            if not (vol_avg and volumes[-1] > vol_avg * 1.3):
                continue
            try:
                rsi_info = data.get_indicator(symbol, "RSI14", "1h")
                rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50
                if rsi is None:
                    rsi = 50
            except Exception:
                rsi = 50
            if rsi > 78:
                continue
            if middle is not None and closes[-1] < middle:
                continue
            return Decision(operation="buy", symbol=symbol, reason="boll breakout", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.06, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

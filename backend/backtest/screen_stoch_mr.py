class Strategy:
    def should_trade(self, data):
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
            klines = data.get_klines(symbol, "4h", 60)
            if len(klines) < 30:
                continue
            closes = [k.close for k in klines]
            lows = [k.low for k in klines]
            highs = [k.high for k in klines]
            ll = min(lows[-14:])
            hh = max(highs[-14:])
            if hh - ll <= 0:
                continue
            kperc = 100 * (closes[-1] - ll) / (hh - ll)
            if kperc > 20:
                continue
            if closes[-1] >= closes[-2]:
                pass
            rc = closes[-15:]
            gains = 0.0
            losses = 0.0
            for i in range(1, len(rc)):
                d = rc[i] - rc[i-1]
                if d > 0:
                    gains += d
                else:
                    losses -= d
            if gains + losses <= 0:
                rsi = 50.0
            else:
                rs = gains / (losses if losses > 0 else 1e-9)
                rsi = 100 - 100 / (1 + rs)
            if rsi > 45:
                continue
            return Decision(operation="buy", symbol=symbol, reason="stoch-mr", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.08, stop_loss_price=price*0.97)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

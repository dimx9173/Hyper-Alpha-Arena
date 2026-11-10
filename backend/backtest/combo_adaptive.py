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
        for symbol in ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR", "SUI", "DOGE", "ARB", "LINK", "UNI", "AAVE", "TAO", "TRUMP"]:
            if symbol in skip:
                continue
            price = data.get_price(symbol)
            if price == 0:
                continue
            klines = data.get_klines(symbol, "4h", 60)
            if len(klines) < 30:
                continue
            closes = [k.close for k in klines]
            if len(closes) < 55:
                continue
            e20 = sum(closes[-20:]) / 20
            e50 = sum(closes[-50:]) / 50
            e20p = sum(closes[-21:-1]) / 20
            e50p = sum(closes[-51:-1]) / 50
            if e20 <= e50 * 0.998:
                continue
            if e20p > e50p * 1.001:
                continue
            highs = [k.high for k in klines]
            lows = [k.low for k in klines]
            hh = max(highs[-21:-1])
            rng50 = (max(highs[-50:]) - min(lows[-50:])) / closes[-50] if closes[-50] else 1
            mom50 = (closes[-1] - closes[-50]) / closes[-50] if closes[-50] else 0
            if rng50 >= 0.10 and abs(mom50) >= 0.03:
                if closes[-1] < hh * 0.97:
                    continue
                return Decision(operation="buy", symbol=symbol, reason="ema-donch-trend", target_portion_of_balance=0.30, leverage=2, max_price=price*1.005, take_profit_price=price*1.08, stop_loss_price=price*0.97)
            w = closes[-20:]
            mean = sum(w) / 20
            var = sum((c - mean) ** 2 for c in w) / 20
            sd = var ** 0.5 if var > 0 else 0
            if sd <= 0:
                continue
            pos = (price - mean) / sd
            if pos > -1.2:
                continue
            rc = closes[-15:]
            gains = 0.0
            losses = 0.0
            for i in range(1, len(rc)):
                d = rc[i] - rc[i-1]
                if d > 0:
                    gains += d
                else:
                    losses -= d
            rsi = 50.0 if gains + losses <= 0 else 100 - 100 / (1 + gains / (losses if losses > 0 else 1e-9))
            if rsi > 38:
                continue
            return Decision(operation="buy", symbol=symbol, reason="mr-range", target_portion_of_balance=0.10, leverage=1, max_price=price*1.005, take_profit_price=price*1.05, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

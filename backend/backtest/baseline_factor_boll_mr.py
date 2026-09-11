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
            if len(klines) < 25:
                continue
            closes = [k.close for k in klines]
            n = 20
            w = closes[-n:]
            mean = sum(w) / n
            var = sum((c - mean) ** 2 for c in w) / n
            sd = var ** 0.5 if var > 0 else 0
            if sd <= 0:
                continue
            upper = mean + 2 * sd
            lower = mean - 2 * sd
            pos = (price - mean) / sd
            if pos > -1.0:
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
            if gains + losses <= 0:
                rsi = 50.0
            else:
                rs = gains / (losses if losses > 0 else 1e-9)
                rsi = 100 - 100 / (1 + rs)
            if rsi > 40:
                continue
            volumes = [k.volume for k in klines]
            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0
            if vol_avg and volumes[-1] > vol_avg * 2.0:
                continue
            return Decision(operation="buy", symbol=symbol, reason="boll-mr oversold", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.08, stop_loss_price=price*0.97)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

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
            klines = data.get_klines(symbol, "1h", 40)
            if len(klines) < 25:
                continue
            closes = [k.close for k in klines]
            n = 20
            w = closes[-n:]
            mean = sum(w) / n
            var = sum((c - mean) ** 2 for c in w) / n
            sd = var ** 0.5 if var > 0 else 0
            upper = mean + 2 * sd
            lower = mean - 2 * sd
            if price <= 0:
                continue
            bw = (upper - lower) / price
            if bw < 0.015 or bw > 0.25:
                continue
            if closes[-1] <= upper:
                continue
            volumes = [k.volume for k in klines]
            vol_avg = sum(volumes[-20:]) / 20 if volumes else 0
            if not (vol_avg and volumes[-1] > vol_avg * 1.15):
                continue
            if closes[-1] < mean:
                continue
            return Decision(operation="buy", symbol=symbol, reason="boll-fast breakout", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.06, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

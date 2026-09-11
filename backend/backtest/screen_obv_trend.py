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
            klines = data.get_klines(symbol, "1h", 60)
            if len(klines) < 30:
                continue
            closes = [k.close for k in klines]
            volumes = [k.volume for k in klines]
            obv = 0.0
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
            obv_prev = 0.0
            for i in range(1, len(closes)-5):
                if closes[i] > closes[i-1]:
                    obv_prev += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv_prev -= volumes[i]
            if obv <= obv_prev:
                continue
            mean20 = sum(closes[-20:]) / 20
            if closes[-1] <= mean20:
                continue
            return Decision(operation="buy", symbol=symbol, reason="obv-trend", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.06, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

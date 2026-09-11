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
            n = 20
            w = closes[-n:]
            mean = sum(w) / n
            var = sum((c - mean) ** 2 for c in w) / n
            sd = var ** 0.5 if var > 0 else 0
            if sd <= 0:
                continue
            pos = (price - mean) / sd
            if pos > -0.8:
                continue
            try:
                md = data.get_market_data(symbol)
                fr = md.get('funding_rate', 0) if isinstance(md, dict) else 0
            except Exception:
                fr = 0
            try:
                fr = float(fr)
            except Exception:
                fr = 0
            if fr > 0.0005:
                continue
            return Decision(operation="buy", symbol=symbol, reason="funding-dip", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.08, stop_loss_price=price*0.97)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

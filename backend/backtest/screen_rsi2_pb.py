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
            if len(closes) < 55:
                continue
            e50 = sum(closes[-50:]) / 50
            if closes[-1] <= e50:
                continue
            d1 = closes[-1] - closes[-2]
            d2 = closes[-2] - closes[-3]
            up = (d1 if d1 > 0 else 0) + (d2 if d2 > 0 else 0)
            dn = (-d1 if d1 < 0 else 0) + (-d2 if d2 < 0 else 0)
            if up + dn <= 0:
                rsi2 = 50.0
            else:
                rs2 = up / (dn if dn > 0 else 1e-9)
                rsi2 = 100 - 100 / (1 + rs2)
            if rsi2 > 25:
                continue
            return Decision(operation="buy", symbol=symbol, reason="rsi2-pullback", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.05, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

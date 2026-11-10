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
        for symbol in ["BTC", "ETH", "SOL", "HYPE", "BNB", "AVAX", "NEAR", "SUI", "DOGE", "ARB", "AAVE", "TAO"]:
            if symbol in skip:
                continue
            price = data.get_price(symbol)
            if price == 0:
                continue
            klines = data.get_klines(symbol, "1h", 80)
            if len(klines) < 60:
                continue
            closes = [k.close for k in klines]
            if len(closes) < 55:
                continue
            e20 = sum(closes[-20:]) / 20
            e50 = sum(closes[-50:]) / 50
            e20p = sum(closes[-21:-1]) / 20
            e50p = sum(closes[-51:-1]) / 50
            if e20 <= e50 * 0.9985:
                continue
            if e20p > e50p * 1.0008:
                continue
            highs = [k.high for k in klines]
            hh = max(highs[-21:-1])
            if closes[-1] < hh * 0.98:
                continue
            return Decision(operation="buy", symbol=symbol, reason="ema-donch-1h", target_portion_of_balance=0.25, leverage=2, max_price=price*1.004, take_profit_price=price*1.06, stop_loss_price=price*0.98, sl_execution="market")
        return Decision(operation="hold", symbol="BTC", reason="no signal")

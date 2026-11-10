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
            volumes = [k.volume for k in klines]
            tpv = sum(((highs[i]+lows[i]+closes[i])/3) * volumes[i] for i in range(len(klines)))
            vv = sum(volumes)
            if vv <= 0:
                continue
            if closes[-1] <= tpv / vv:
                continue
            return Decision(operation="buy", symbol=symbol, reason="ema-vwap", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.08, stop_loss_price=price*0.97)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

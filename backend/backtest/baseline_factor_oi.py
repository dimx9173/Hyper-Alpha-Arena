class Strategy:
    def should_trade(self, data):
        # factor-D: OI expansion + taker + 1h trend, lev2, TP6%/SL2.5%, max3 no-add
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
            try:
                oi = data.get_flow(symbol, "OI_DELTA", "1h")
                oi_chg = oi.get("current", 0) if isinstance(oi, dict) else 0
            except Exception:
                oi_chg = 0
            if oi_chg is None or oi_chg <= 1.0:
                continue
            try:
                tk = data.get_flow(symbol, "TAKER", "1h")
                ratio = tk.get("current", 1) if isinstance(tk, dict) else 1
                if isinstance(ratio, dict):
                    ratio = ratio.get("taker_ratio", 1)
            except Exception:
                ratio = 1
            try:
                ratio = float(ratio)
            except Exception:
                ratio = 1
            if ratio <= 1.02:
                continue
            klines = data.get_klines(symbol, "1h", 80)
            if len(klines) < 50:
                continue
            closes = [k.close for k in klines]
            ma_fast = sum(closes[-8:]) / 8
            ma_slow = sum(closes[-30:]) / 30
            if ma_slow <= 0:
                continue
            if (ma_fast - ma_slow) / ma_slow < 0.004:
                continue
            if closes[-1] <= ma_fast:
                continue
            try:
                rsi_info = data.get_indicator(symbol, "RSI14", "1h")
                rsi = rsi_info.get("value", 50) if isinstance(rsi_info, dict) else 50
                if rsi is None:
                    rsi = 50
            except Exception:
                rsi = 50
            if rsi > 75 or rsi < 48:
                continue
            return Decision(operation="buy", symbol=symbol, reason="oi+taker trend", target_portion_of_balance=0.25, leverage=2, max_price=price*1.005, take_profit_price=price*1.06, stop_loss_price=price*0.975)
        return Decision(operation="hold", symbol="BTC", reason="no signal")

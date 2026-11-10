#!/usr/bin/env python3
"""
Fetch kline data from Hyperliquid API and persist to PostgreSQL.
Populates crypto_klines table for GA backtest.
"""
import sys, time, os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.chdir(_HERE)

from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from services.hyperliquid_market_data import HyperliquidClient
from database.models import CryptoKline

DB_URL = os.environ.get("DATABASE_URL", "postgresql://alpha_user:alpha_pass@/alpha_arena?host=/tmp")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

SYMBOLS = ["BTC", "ETH", "SOL", "HYPE", "DOGE", "SUI", "ARB", "AVAX", "BNB", "NEAR", "AAVE", "TAO"]
PERIODS = ["5m", "15m", "1h", "4h", "1d"]
EXCHANGE = "hyperliquid"

NOW_MS = int(time.time() * 1000)
SINCE_MS = NOW_MS - (365 * 24 * 3600 * 1000)

client = HyperliquidClient(environment="mainnet")

def main():
    db = SessionLocal()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM crypto_klines")).scalar()
        print(f"Current crypto_klines count: {total}")

        for symbol in SYMBOLS:
            print(f"\n=== {symbol} ===", flush=True)
            for period in PERIODS:
                existing = db.execute(text("""
                    SELECT COUNT(*) FROM crypto_klines 
                    WHERE exchange = :ex AND symbol = :sym AND period = :per
                """), {"ex": EXCHANGE, "sym": symbol, "per": period}).scalar()
                if existing > 100 and not (period == "1h" and existing < 8000):
                    print(f"  {period}: existing={existing} (skipping)", flush=True)
                    continue
                if period == "1h" and existing > 100:
                    print(f"  {period}: existing={existing}, paging back to fill year...", end=" ", flush=True)
                else:
                    print(f"  {period}: fetching...", end=" ", flush=True)
                klines = client._fetch_kline_native_range(symbol, period, SINCE_MS, NOW_MS, persist=False)
                # HL candleSnapshot 單次上限 5000 根：1h 一年要 8760 根，一次抓不完。
                # 若回來的第一根晚於 SINCE_MS，分段往前補，直到覆蓋或無更多資料。
                for _ in range(4):
                    if not klines or klines[0]["timestamp"] * 1000 <= SINCE_MS:
                        break
                    first_ms = klines[0]["timestamp"] * 1000
                    more = client._fetch_kline_native_range(symbol, period, SINCE_MS, first_ms - 1, persist=False)
                    if not more:
                        break
                    seen = {k["timestamp"] for k in klines}
                    klines = [k for k in more if k["timestamp"] not in seen] + klines
                    if len(more) < 100:
                        break
                if not klines:
                    print("(no data)")
                    continue
                inserted = 0
                for k in klines:
                    ts = k["timestamp"]
                    existing = db.execute(text("""
                        SELECT 1 FROM crypto_klines 
                        WHERE exchange = :ex AND symbol = :sym AND period = :per AND timestamp = :ts
                    """), {"ex": EXCHANGE, "sym": symbol, "per": period, "ts": ts}).fetchone()
                    if existing:
                        continue
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    record = CryptoKline(
                        exchange=EXCHANGE, symbol=symbol, market="CRYPTO",
                        period=period, timestamp=ts, datetime_str=dt.strftime("%Y-%m-%d %H:%M:%S"),
                        environment="mainnet",
                        open_price=k["open"], high_price=k["high"], low_price=k["low"],
                        close_price=k["close"], volume=k["volume"],
                    )
                    db.add(record)
                    inserted += 1
                if inserted > 0:
                    db.commit()
                print(f"inserted={inserted} (total fetched={len(klines)})")
                time.sleep(0.2)

        total = db.execute(text("SELECT COUNT(*) FROM crypto_klines")).scalar()
        print(f"\n=== FINAL === Total crypto_klines: {total}")
    finally:
        db.close()

if __name__ == "__main__":
    main()

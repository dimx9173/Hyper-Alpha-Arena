#!/usr/bin/env python3
"""Backfill three years of Binance USDT-M 1h klines for the top 30 symbols."""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone

import requests

from database.connection import SessionLocal
from services.exchanges.binance_adapter import BinanceAdapter
from services.exchanges.data_persistence import ExchangeDataPersistence

EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
TICKER_24H = "https://fapi.binance.com/fapi/v1/ticker/24hr"
STABLES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDT", "USDE", "USTC", "USDS", "EURI",
}
INTERVAL_MS = 60 * 60 * 1000


def top_symbols(limit: int = 30) -> list[str]:
    info = requests.get(EXCHANGE_INFO, timeout=20)
    info.raise_for_status()
    tickers = requests.get(TICKER_24H, timeout=20)
    tickers.raise_for_status()
    valid = {
        row["symbol"]: row["baseAsset"]
        for row in info.json().get("symbols", [])
        if row.get("status") == "TRADING"
        and row.get("quoteAsset") == "USDT"
        and row.get("contractType") == "PERPETUAL"
    }
    ranked = []
    for row in tickers.json():
        pair = row.get("symbol")
        base = valid.get(pair)
        if not base or base in STABLES or not re.fullmatch(r"[A-Z0-9]{2,20}", base):
            continue
        try:
            volume = float(row.get("quoteVolume") or 0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume > 0:
            ranked.append((volume, base))
    ranked.sort(reverse=True)
    return [base for _, base in ranked[:limit]]


def backfill_symbol(symbol: str, start_ms: int, end_ms: int, adapter: BinanceAdapter, db) -> dict:
    persistence = ExchangeDataPersistence(db)
    cursor = start_ms
    batches = inserted = updated = 0
    while cursor < end_ms:
        klines = adapter.fetch_klines(
            symbol, "1h", limit=1500, start_time=cursor, end_time=end_ms
        )
        if not klines:
            break
        result = persistence.save_klines(klines, environment="mainnet")
        inserted += result["inserted"]
        updated += result["updated"]
        batches += 1
        last_ms = int(klines[-1].timestamp) * 1000 + INTERVAL_MS
        next_cursor = max(cursor + INTERVAL_MS, last_ms)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.15)
    return {"symbol": symbol, "batches": batches, "inserted": inserted, "updated": updated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - args.days * 24 * 60 * 60 * 1000
    symbols = top_symbols(args.limit)
    print(json.dumps({
        "exchange": "binance", "environment": "mainnet", "period": "1h",
        "days": args.days, "start": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat(),
        "symbols": symbols, "dry_run": args.dry_run,
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    adapter = BinanceAdapter(environment="mainnet")
    db = SessionLocal()
    results = []
    try:
        for index, symbol in enumerate(symbols, 1):
            item = backfill_symbol(symbol, start_ms, now_ms, adapter, db)
            results.append(item)
            print(json.dumps({"progress": f"{index}/{len(symbols)}", **item}), flush=True)
    finally:
        db.close()
    print(json.dumps({"completed": len(results), "results": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

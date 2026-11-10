"""
Hyperliquid 180日 K線回填腳本 — Top30（剔除穩定幣）全量補齊

用法：
  PYTHONPATH=backend DATABASE_URL=... backend/.venv/bin/python -m scripts.backfill_hl_180d --top30 30 --days 180 --periods 5m,15m,1h --dry-run
  PYTHONPATH=backend DATABASE_URL=... backend/.venv/bin/python -m scripts.backfill_hl_180d --top30 30 --days 180 --periods 5m,15m,1h,4h,1d
  PYTHONPATH=backend DATABASE_URL=... backend/.venv/bin/python -m scripts.backfill_hl_180d --symbols BTC,ETH,SOL --days 30 --periods 5m
"""
import argparse
import logging
import sys
import time
from typing import List

logger = logging.getLogger(__name__)

# 全局穩定幣黑名單（與 hl_universe 保持一致）
STABLES = {"USDC", "USDT", "USDS", "USDH", "FEUSD", "USDE", "USDY", "USDD", "DAI", "TUSD", "BUSD", "FRAX", "GUSD", "PYUSD", "USDP", "EURI", "EURC", "USDD", "USDE"}

_FALLBACK_TOP30 = [
    "BTC", "ETH", "HYPE", "SOL", "ZEC", "PUMP", "XRP", "LIT", "XMR", "PONS",
    "SKR", "kPEPE", "ENA", "UNI", "TRUMP", "NEAR", "FARTCOIN", "CASHCAT", "XPL", "TAO",
    "GRAM", "SUI", "DOGE", "LINK", "CRV", "AAVE", "ARB", "ADA", "BNB", "ZORA",
]

PERIOD_SEC = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
}


def _resolve_symbols(args) -> List[str]:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return [s for s in syms if s not in STABLES]
    if args.top30:
        try:
            try:
                from backtest.hl_universe import get_hl_top30
            except ImportError:
                from backend.backtest.hl_universe import get_hl_top30
            syms = get_hl_top30(limit=args.top30, exclude_stables=True)
            syms = [s for s in syms if s not in STABLES]
            if syms:
                logger.info(f"動態 Top{args.top30}：{syms}")
                return syms
        except Exception as e:
            logger.warning(f"get_hl_top30 失敗 fallback: {e}")
        logger.warning(f"Top30 發現失敗，fallback 到內建 {_FALLBACK_TOP30[:args.top30]}")
        return [s for s in _FALLBACK_TOP30[: args.top30] if s not in STABLES]
    # 預設 Top30
    return [s for s in _FALLBACK_TOP30[:30] if s not in STABLES]


def _expected_count(days: int, period: str) -> int:
    sec = PERIOD_SEC.get(period, 300)
    return int(days * 86400 / sec)


def main():
    parser = argparse.ArgumentParser(description="Hyperliquid 180日 K線回填 — Top30 全量補齊（剔除穩定幣）")
    parser.add_argument("--symbols", type=str, default=None, help="逗號分隔，如 BTC,ETH,SOL（與 --top30 互斥）")
    parser.add_argument("--top30", type=int, default=30, help="動態發現 TopN（預設30，0則用 --symbols）")
    parser.add_argument("--days", type=int, default=180, help="回填天數（預設180）")
    parser.add_argument("--periods", type=str, default="5m,15m,1h,4h,1d", help="逗號分隔週期，預設 5m,15m,1h,4h,1d")
    parser.add_argument("--exchange", type=str, default="hyperliquid", choices=["hyperliquid", "binance"], help="交易所")
    parser.add_argument("--environment", type=str, default="mainnet", help="環境 mainnet/testnet")
    parser.add_argument("--batch-size", type=int, default=500, help="單次拉取上限（HL 單次 500）")
    parser.add_argument("--sleep", type=float, default=0.8, help="請求間隔秒（限流，預設 0.8 避免 429）")
    parser.add_argument("--dry-run", action="store_true", help="僅列出將回填的幣種與區間，不寫庫")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.symbols and args.top30 and args.top30 != 30:
        # 明確指定 symbols 時忽略 top30
        pass

    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    symbols = _resolve_symbols(args) if not args.symbols else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # 穩定幣二次過濾
    symbols = [s for s in symbols if s not in STABLES]
    if not symbols:
        logger.error("無可用幣種（全被穩定幣過濾）")
        sys.exit(1)

    now_ms = int(time.time() * 1000)
    now_s = now_ms // 1000
    start_s = now_s - args.days * 86400

    logger.info(f"回填配置：exchange={args.exchange} env={args.environment} days={args.days} periods={periods} symbols={len(symbols)} sleep={args.sleep}s")
    for sym in symbols:
        logger.info(f"  - {sym}")

    if args.dry_run:
        for sym in symbols:
            for per in periods:
                exp = _expected_count(args.days, per)
                logger.info(f"[dry-run] {sym} {per}: {args.days}d 約 {exp} 根，區間 {start_s} → {now_s}")
        logger.info(f"[dry-run] 完成：{len(symbols)} 幣種 × {len(periods)} 週期 × {args.days}d，預估總計 {sum(_expected_count(args.days, p) for p in periods) * len(symbols)} 根")
        return

    # 真回填
    try:
        from database.connection import SessionLocal
    except ImportError:
        from backend.database.connection import SessionLocal
    from repositories.kline_repo import KlineRepository

    # 延遲導入 HL 客戶端（僅 hyperliquid 需要）
    # 注意：CCXT hyperliquid 的 fetch_markets 在當前版本有 spot 解析崩潰（None+str），
    # 而原生 POST /info candleSnapshot 已驗證正常（BTC 5m 2017根），故直接走原生管線。
    _hl_client = None
    _hl_fetch = None  # type: ignore
    if args.exchange == "hyperliquid":
        try:
            try:
                from services.hyperliquid_market_data import HyperliquidClient
            except ImportError:
                from backend.services.hyperliquid_market_data import HyperliquidClient  # type: ignore
            _hl_client = HyperliquidClient(environment=args.environment)

            def _native_fetch(symbol: str, period: str, since_ms: int, until_ms: int, environment: str = "mainnet"):  # type: ignore
                # 直接走原生 candleSnapshot，繞過 CCXT
                return _hl_client._fetch_kline_native_range(symbol, period, since_ms, until_ms)  # type: ignore

            _hl_fetch = _native_fetch  # type: ignore
        except Exception as e:
            logger.warning(f"初始化原生 HL 客戶端失敗，嘗試 fallback 到 get_historical_kline_data: {e}")
            try:
                from services.hyperliquid_market_data import get_historical_kline_data_from_hyperliquid
                _hl_fetch = get_historical_kline_data_from_hyperliquid  # type: ignore
            except ImportError:
                from backend.services.hyperliquid_market_data import get_historical_kline_data_from_hyperliquid as _hl_fetch  # type: ignore
    else:
        _hl_fetch = None  # type: ignore

    db = SessionLocal()
    repo = KlineRepository(db)

    total_inserted = 0
    total_failed: List[str] = []
    total_skipped = 0
    t0 = time.time()

    try:
        for sym in symbols:
            for per in periods:
                sec = PERIOD_SEC.get(per, 300)
                expected = _expected_count(args.days, per)
                # 斷點：查 DB 已有數量
                try:
                    from sqlalchemy import text as _text
                    row = db.execute(_text("SELECT count(*) FROM crypto_klines WHERE symbol=:s AND exchange=:ex AND period=:p AND timestamp BETWEEN :a AND :b"), {"s": sym, "ex": args.exchange, "p": per, "a": start_s, "b": now_s}).fetchone()
                    cnt = int(row[0]) if row else 0
                except Exception:
                    cnt = 0

                # 若已接近完整（>95%）則跳過
                if cnt >= expected * 0.95 and expected > 0:
                    logger.info(f"[skip] {sym} {per}: 已有 {cnt}/{expected} 根（{cnt/expected*100:.1f}%），跳過")
                    total_skipped += 1
                    continue

                # 找缺口
                try:
                    gaps = repo.get_missing_ranges(args.exchange, sym, per, start_s, now_s, environment=args.environment)
                except Exception as e:
                    logger.warning(f"get_missing_ranges 失敗 {sym} {per}: {e}，改為全量拉取")
                    gaps = [(start_s, now_s)]

                if not gaps:
                    logger.info(f"[skip] {sym} {per}: 無缺口")
                    total_skipped += 1
                    continue

                for gap_start, gap_end in gaps:
                    # 分段拉取（避免單次超限）
                    cur = gap_start * 1000
                    end_ms = gap_end * 1000
                    while cur <= end_ms:
                        batch_since = cur
                        batch_until = min(cur + args.batch_size * sec * 1000, end_ms)
                        # 取數帶 429 退避重試
                        klines = None
                        for attempt in range(5):
                            try:
                                if args.exchange == "hyperliquid":
                                    klines = _hl_fetch(symbol=sym, period=per, since_ms=batch_since, until_ms=batch_until, environment=args.environment)
                                else:
                                    logger.warning(f"Binance 回填請用 HistoricalDataProvider 觸發：{sym} {per}")
                                    klines = []
                                break
                            except Exception as e:
                                msg = str(e)
                                if "429" in msg or "Too Many Requests" in msg:
                                    backoff = 2 * (attempt + 1)
                                    logger.warning(f"429 限流 {sym} {per} [{batch_since}-{batch_until}] attempt {attempt+1}/5，退避 {backoff}s: {e}")
                                    time.sleep(backoff)
                                    continue
                                logger.warning(f"拉取失敗 {sym} {per} [{batch_since}-{batch_until}]: {e}")
                                klines = []
                                total_failed.append(f"{sym}/{per}")
                                break
                        if klines is None:
                            klines = []
                            total_failed.append(f"{sym}/{per}")

                        if klines:
                            try:
                                res = repo.save_kline_data(symbol=sym, market="CRYPTO", period=per, kline_data=klines, exchange=args.exchange, environment=args.environment)
                                total_inserted += res.get("inserted", 0)
                                logger.info(f"[ok] {sym} {per} [{batch_since//1000}-{batch_until//1000}] 寫入 inserted={res.get('inserted')} updated={res.get('updated')} total={len(klines)}")
                            except Exception as e:
                                # 唯一鍵衝突會毒化事務，必須 rollback 否則後續全崩
                                try:
                                    db.rollback()
                                except Exception:
                                    pass
                                logger.warning(f"save 失敗 {sym} {per} [{batch_since//1000}-{batch_until//1000}]: {e}")
                        else:
                            logger.debug(f"[empty] {sym} {per} [{batch_since}-{batch_until}] 無數據")

                        # 限流
                        time.sleep(args.sleep)
                        # 推進
                        if klines:
                            # 以最後一根 timestamp 推進
                            try:
                                last_ts = klines[-1].get("timestamp", batch_until // 1000)
                                cur = (last_ts + sec) * 1000
                            except Exception:
                                cur = batch_until + sec * 1000
                        else:
                            cur = batch_until + sec * 1000

                        # 若已覆蓋全缺口則結束
                        if cur > end_ms:
                            break

        elapsed = time.time() - t0
        logger.info(f"回填完成：inserted={total_inserted} skipped={total_skipped} failed={len(total_failed)} elapsed={elapsed:.1f}s")
        if total_failed:
            logger.warning(f"失敗項：{total_failed[:20]}")
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

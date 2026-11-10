"""
Unified Iteration Entry Point — GA + Annual (365d) Backtest as Default
=======================================================================

這是預設的策略迭代入口，強制：
1. 使用基因演算法 (GA) 作為優化引擎
2. 使用 365 天以上的歷史資料進行回測
3. 以 Annual Gate (365d validator) 為主要驗證標準

用法:
    PYTHONPATH=backend python -m backtest.iterate [options]

主要選項:
    --generations N       GA 世代數 (預設 25)
    --pop-size N          族群大小 (預設 20)
    --workers N           平行 workers (預設 4)
    --elite N             精英保留數 (預設 2)
    --tournament N        錦標賽大小 (預設 3)
    --cx-rate RATE        交配率 (預設 0.7)
    --mut-rate RATE       突變率 (預設 0.2)
    --early-stop N        連續無改善停止 (預設 8)
    --resume              續跑
    --seed N              隨機種子
    --exchange EXCH       交易所 (預設 hyperliquid)
    --symbols SYM...      指定幣種 (預設 HL Top30)
    --no-validate         跳過驗證 (不建議)

輸出:
    - best_code_annual.py     最終最佳策略代碼
    - summary_annual.json     年度回測摘要
    - iteration_log.jsonl     迭代日誌
"""

import argparse
import logging
import sys
import pathlib

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Hyper-Alpha-Arena 策略迭代 — GA + Annual (365d) 強制模式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # GA 參數
    ga_group = parser.add_argument_group("Genetic Algorithm Parameters")
    ga_group.add_argument("--generations", type=int, default=25, help="GA 世代數")
    ga_group.add_argument("--pop-size", type=int, default=20, help="族群大小")
    ga_group.add_argument("--workers", type=int, default=4, help="平行 workers 數")
    ga_group.add_argument("--elite", type=int, default=2, help="精英保留數")
    ga_group.add_argument("--tournament", type=int, default=3, help="錦標賽大小")
    ga_group.add_argument("--cx-rate", type=float, default=0.7, help="交配率")
    ga_group.add_argument("--mut-rate", type=float, default=0.2, help="突變率")
    ga_group.add_argument("--early-stop", type=int, default=8, help="連續無改善停止代數")

    # 年度回測參數（強制 365d）
    annual_group = parser.add_argument_group("Annual Backtest Parameters (FORCED 365d+)")
    annual_group.add_argument(
        "--min-days",
        type=int,
        default=365,
        help="最少回測天數 (強制 365，不可降低)",
    )
    annual_group.add_argument(
        "--target-pnl-pct",
        type=float,
        default=120.0,
        help="目標年化報酬 % (月均 10% = 120%/年)",
    )
    annual_group.add_argument(
        "--max-maxdd-pct",
        type=float,
        default=5.0,
        help="最大回撤 % 上限",
    )

    # 其他參數
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--resume", action="store_true", help="從中斷處續跑")
    other_group.add_argument("--seed", type=int, default=None, help="隨機種子")
    other_group.add_argument("--exchange", type=str, default="hyperliquid", help="交易所")
    other_group.add_argument(
        "--symbols", nargs="+", default=None, help="指定幣種 (預設 HL Top30)"
    )
    other_group.add_argument(
        "--no-validate",
        action="store_true",
        help="跳過年度驗證 (不建議，僅開發用)",
    )
    other_group.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="輸出目錄",
    )

    args = parser.parse_args()

    # 強制檢查：最少 365 天
    if args.min_days < 365:
        logger.error(f"ERROR: min-days 必須 >= 365，目前設定: {args.min_days}")
        logger.error("本系統強制使用年度級回測 (365+ 天) 以確保策略穩健性")
        sys.exit(1)

    # 動態導入 GA 迭代器
    try:
        from backtest.iterate_annual_ga import run_ga_iteration
        from backtest.ab_validators import annual_validator
    except ImportError as e:
        logger.error(f"無法導入模組: {e}")
        logger.error("請確認 PYTHONPATH 包含 backend 目錄")
        sys.exit(1)

    # 顯示配置摘要
    logger.info("=" * 60)
    logger.info("Hyper-Alpha-Arena 策略迭代 — GA + Annual (365d) 模式")
    logger.info("=" * 60)
    logger.info(f"GA 參數: generations={args.generations}, pop_size={args.pop_size}, "
                f"elite={args.elite}, tournament={args.tournament}, "
                f"cx_rate={args.cx_rate}, mut_rate={args.mut_rate}")
    logger.info(f"回測週期: 強制 {args.min_days} 天 (年度級)")
    logger.info(f"驗證標準: Annual Gate — pnl%>={args.target_pnl_pct}%, "
                f"maxDD%<={args.max_maxdd_pct}%, PF>1.5, Sharpe>0.5, Win>45%")
    logger.info(f"交易所: {args.exchange}")
    logger.info(f"輸出目錄: {args.output_dir}")
    if args.resume:
        logger.info("模式: 續跑模式")
    logger.info("=" * 60)

    # 執行 GA 迭代
    try:
        result = run_ga_iteration(
            generations=args.generations,
            pop_size=args.pop_size,
            workers=args.workers,
            elite=args.elite,
            tournament=args.tournament,
            cx_rate=args.cx_rate,
            mut_rate=args.mut_rate,
            early_stop=args.early_stop,
            min_days=args.min_days,
            target_pnl_pct=args.target_pnl_pct,
            max_maxdd_pct=args.max_maxdd_pct,
            resume=args.resume,
            seed=args.seed,
            exchange=args.exchange,
            symbols=args.symbols,
            skip_validation=args.no_validate,
            output_dir=args.output_dir,
        )

        if result:
            logger.info("=" * 60)
            logger.info("迭代完成！最佳策略已輸出")
            logger.info(f"最佳適應度: {result.get('best_fitness', 'N/A')}")
            logger.info(f"年化報酬: {result.get('best_pnl_365d', 'N/A')}%")
            logger.info(f"最大回撤: {result.get('best_maxdd_365d', 'N/A')}%")
            logger.info(f"年化 Sharpe: {result.get('best_sharpe_365d', 'N/A')}")
            logger.info("=" * 60)
            sys.exit(0)
        else:
            logger.error("迭代失敗：未找到通過 Annual Gate 的策略")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("使用者中斷，進度已儲存，可使用 --resume 續跑")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"迭代過程發生錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
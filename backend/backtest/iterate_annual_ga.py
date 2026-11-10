#!/usr/bin/env python3
"""
GA Annual Strategy Iteration CLI

真正的基因演算法：族群並行、錦標賽選擇、交配、突變、精英保留
"""

import argparse
import json
import random
from copy import deepcopy
import logging
import pathlib
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# Add backend to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backtest.ga_annual import (
    Genome, random_genome, clamp, calculate_fitness, set_exploration_gens,
    tournament_select, crossover, mutate, population_diversity,
    render_strategy, evaluate_individual, BOUNDS
)

logger = logging.getLogger(__name__)


class _SeedResult:
    """只為了把日誌欄位餵給 gate_violation（它讀 pnl_365d 等屬性）。"""
    def __init__(self, rec):
        self.pnl_365d = float(rec.get("pnl_365d", 0.0) or 0.0)
        self.maxdd_365d = float(rec.get("maxdd_365d", 100.0) or 100.0)
        self.sharpe_365d = float(rec.get("sharpe_365d", 0.0) or 0.0)
        self.pf = rec.get("pf")
        self.trades_365d = int(rec.get("trades_365d", 0) or 0)
        self.win_rate = float(rec.get("win_rate", 0.0) or 0.0)


def _strict_violation(rec: Dict[str, Any]) -> float:
    """重算嚴格 gate 違規量；失敗時回傳保守的 1.0。"""
    try:
        from backtest.ga_annual import gate_violation
        return float(gate_violation(_SeedResult(rec)))
    except Exception:
        return 1.0


def _load_seed_genomes(seed_log: Optional[str], pop_size: int, logger) -> List[Genome]:
    """從先前的 iteration_log jsonl 載入種子基因組。

    只取通過 gate 的個體，依 fitness 由高到低排序，並以 genome_hash 去重。
    理由：GA 的瓶頸一直是「找不到嚴格可行域」，而不是在可行域裡不夠好。
    已知可行解能讓族群一開始就有嚴格通過者，後續只需往上改進。
    """
    if not seed_log:
        return []
    path = Path(seed_log)
    if not path.exists():
        logger.warning(f"seed log not found, falling back to random: {seed_log}")
        return []
    try:
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "individual" or not rec.get("gate_annual"):
                    continue
                if not rec.get("genome"):
                    continue
                rows.append(rec)
        # 排序鍵：先「嚴格 gate_violation == 0」，再依 fitness 由高到低。
        # 日誌的 gate_annual 在探索期反映的是放寬 gate，直接信任會混入只是
        # 探索通過、實際未達嚴格門檻的個體（例如本專案 smoke 的 24 個通過者中
        # 有 2 個屬於此類），所以在這裡重算一次嚴格違規量。
        for rec in rows:
            rec["_viol"] = _strict_violation(rec)
        rows.sort(key=lambda r: (0 if r["_viol"] < 1e-9 else 1, -float(r.get("fitness", 0.0))))
        n_strict = sum(1 for r in rows if r["_viol"] < 1e-9)
        seeds, seen = [], set()
        for rec in rows:
            h = rec.get("genome_hash")
            if h and h in seen:
                continue
            if h:
                seen.add(h)
            # 只在「一個嚴格可行種子都沒有」時才允許用探索通過者補位，
            # 否則就是拿雜訊稀釋已經確認可行的起始族群。
            if n_strict > 0 and rec["_viol"] >= 1e-9:
                continue
            try:
                seeds.append(Genome.from_dict(rec["genome"]))
            except Exception:
                continue
            if len(seeds) >= pop_size:
                break
        logger.info(f"seed log {seed_log}: {len(rows)} gate-passing individuals "
                    f"({n_strict} strictly feasible), seeded {len(seeds)} strictly-feasible genome(s)")
        return seeds
    except Exception as e:
        logger.warning(f"seed log read failed ({e}), falling back to random")
        return []


def main():
    parser = argparse.ArgumentParser(description="GA Annual (365d) strategy iteration")
    parser.add_argument("--generations", type=int, default=25, help="number of generations")
    parser.add_argument("--pop-size", type=int, default=20, help="population size")
    parser.add_argument("--elite", type=int, default=2, help="elite count")
    parser.add_argument("--tournament", type=int, default=3, help="tournament size")
    parser.add_argument("--cx-rate", type=float, default=0.8, help="crossover rate")
    parser.add_argument("--mut-rate", type=float, default=0.2, help="mutation rate per gene")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers")
    parser.add_argument("--exchange", type=str, default="hyperliquid", choices=["binance", "hyperliquid"])
    parser.add_argument("--symbols", type=str, default="BTC,ETH,SOL,HYPE,DOGE,SUI,ARB,AVAX,BNB,NEAR,AAVE,TAO")
    parser.add_argument("--log-file", type=str, default=None, help="log file path")
    parser.add_argument("--resume", action="store_true", help="resume from existing log")
    parser.add_argument("--best-code", type=str, default=None, help="output best code path")
    parser.add_argument("--summary", type=str, default=None, help="output summary path")
    parser.add_argument("--early-stop", type=int, default=5,
                        help="early stop if no improvement for N generations. "
                             "0 disables early stopping entirely (runs all --generations). "
                             "Note: this is a threshold on the improvement counter, so "
                             "any value <= 0 stops after generation 1 unless handled specially.")
    parser.add_argument("--windows", type=int, default=1,
                        help="切成幾個互不重疊的視窗分別評估。1 = 舊的單一視窗行為。"
                             "2 或 3 可避免單一強行情主導 fitness。")
    parser.add_argument("--window-recent-weight", type=float, default=1.5,
                        help="最近視窗的權重，其餘視窗為 1.0。")
    parser.add_argument("--end-time-ms", type=int, default=0,
                        help="固定回測視窗終點（epoch ms）。0 = 啟動時取一次當下時間並沿用整輪。"
                             "固定視窗才能讓同一基因組在每個世代得到相同結果，"
                             "否則 GA 會在對時間漂移造成的雜訊最佳化。")
    parser.add_argument("--seed-log", type=str, default=None,
                        help="prior iteration_log jsonl to seed the initial population from; "
                             "prefers individuals that passed the gate, ranked by fitness")
    parser.add_argument("--seed", type=int, default=0,
                        help="固定 random 種子。0 = 啟動時取當下時間（每次都是新搜尋）；"
                             "非 0 = 完全可重現，同一 seed 重跑得到相同初始族群。")
    parser.add_argument("--exploration-gens", type=int, default=None,
                        help="fixed number of leading exploration generations; "
                             "default = max(3, int(generations * 0.5))")
    parser.add_argument("--backtest-days", type=int, default=365, help="backtest window; >365 reports average annual metrics")
    parser.add_argument("--kline-period", type=str, default="1h",
                        help="K 線週期。HL 的 1h 只有約半年整，整年 refit 用 2h。")
    args = parser.parse_args()

    # 決定性：固定 random 種子。
    # 沒有這行時 random.randint/uniform 讀的是 process 開機以來的全域 RNG 狀態，
    # 同一組參數重跑會得到不同的初始族群，GA 結果無法重現也無法回溯比對。
    # seed 為 0 時取當下時間（保持「每次執行都是新的隨機搜尋」的原意），
    # 傳入非 0 則完全可重現。
    _rng_seed = args.seed if args.seed else int(time.time())
    random.seed(_rng_seed)

    # 探索期長度必須在 worker 進程建立前設定，因為 evaluate_individual 會讀取它
    if args.exploration_gens is not None:
        set_exploration_gens(args.exploration_gens)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("random seed = %s", _rng_seed)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    default_log = pathlib.Path(__file__).parent / "iteration_log_annual_ga.jsonl"
    log_file = pathlib.Path(args.log_file) if args.log_file else default_log
    best_code_path = pathlib.Path(args.best_code) if args.best_code else (pathlib.Path(__file__).parent / "best_code_annual_ga.py")
    summary_path = pathlib.Path(args.summary) if args.summary else (pathlib.Path(__file__).parent / "summary_annual_ga.json")

    # Resume logic
    completed_generations = 0
    population = []
    best_fitness = -1000.0
    best_genome = None
    best_gen = 0
    no_improve_count = 0
    all_logs = []

    if log_file.exists() and args.resume:
        try:
            with log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        all_logs.append(rec)
                        if rec.get("type") == "generation_summary":
                            completed_generations = max(completed_generations, rec.get("generation", 0))
                        if rec.get("type") == "best_overall":
                            best_fitness = rec.get("fitness", -1000)
                            best_gen = rec.get("generation", 0)
                            best_genome = Genome.from_dict(rec.get("genome", {}))
                    except Exception:
                        continue
            logger.info(f"resume: {completed_generations} generations logged, best_fitness={best_fitness:.3f}")
        except Exception as e:
            logger.warning(f"resume read failed: {e}")

    # Initialize population
    if completed_generations == 0:
        # Generation 0: seeded passers (if any) + baseline + random
        seeds = _load_seed_genomes(args.seed_log, args.pop_size, logger)
        if seeds:
            logger.info(f"Seeded initial population with {len(seeds)} gate-passing genome(s) "
                        f"from {args.seed_log}")
        # 保留一個預設 baseline 槽位，避免種子品質不佳時整個族群同質化
        population = [Genome()] + seeds
        while len(population) < args.pop_size:
            population.append(random_genome())
        population = population[:args.pop_size]
    else:
        # Load from last generation
        # Find last generation's population
        last_gen_logs = [r for r in all_logs if r.get("type") == "individual" and r.get("generation") == completed_generations]
        if last_gen_logs:
            population = [Genome.from_dict(r["genome"]) for r in last_gen_logs]
        else:
            # Fallback: random
            population = [Genome() for _ in range(args.pop_size)]
            for i in range(args.pop_size):
                population[i] = random_genome()

    logger.info(f"Population size: {len(population)}")

    # Initialize best individual metrics
    best_genome_pnl = 0.0
    best_genome_maxdd = 100.0
    best_genome_sharpe = 0.0

    # 整輪 run 共用同一個回測視窗終點（2026-09-27）。
    # 原本 evaluate_individual 內部各自呼叫 time.time()，同一基因組在不同世代
    # 會得到不同指標（實測 DD 3.69% -> 5.01%），使 GA 無法重現評估。
    end_time_ms = args.end_time_ms if args.end_time_ms else int(time.time() * 1000)

    # 多視窗一致性評分（2026-09-27）。
    # 單一視窗會系統性獎勵「只在強行情賺錢的策略」，因此改成巢狀視窗並加懲罰。
    #
    # 為什麼是「巢狀」（全部結束於 now、長度遞增）而不是「互不重疊」：
    # 互不重疊切分實測失敗。windows=2 + 1095 天切成兩段 547 天時，
    # 較舊那段是 2024-01~2025-03，期間 HYPE/TAKE/PUMP 尚未上市，
    # 可用標的少 -> 交易數天然偏低（實測年化 18.02 < 門檻 20），
    # 該視窗 0/60 通過，一致性懲罰恆為 -1.0。GA 於是被推去最佳化
    # 「在少數標的上能否湊到 20 筆」，這不是本專案的目的。
    # 巢狀視窗讓每一段都含完整的近期資料，門檻才具有可比性。
    windows = [(end_time_ms, args.backtest_days, 1.0)]
    if args.windows > 1:
        windows = []
        for _i in range(args.windows):
            # 長度遞增：第 0 段最短（最近），最後一段是完整的 backtest_days。
            # 全部結束於 end_time_ms；較短的那段拿到較高權重。
            _k = _i + 1
            _days = max(1, args.backtest_days * _k // args.windows)
            _w = args.window_recent_weight if _i == 0 else 1.0
            windows.append((end_time_ms, _days, _w))
    logger.info(f"Fixed backtest window end_time_ms={end_time_ms}  windows={windows}")

    log_file.parent.mkdir(parents=True, exist_ok=True)

    total_generations = args.generations
    start_gen = completed_generations + 1

    gen = 0  # Initialize gen for summary
    for gen in range(start_gen, total_generations + 1):
        gen_start = time.time()
        logger.info(f"=== Generation {gen}/{total_generations} ===")

        # Evaluate population in parallel
        eval_futures = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for ind in population:
                future = executor.submit(evaluate_individual, ind.to_dict(), symbols, args.exchange, gen, args.generations, args.backtest_days, end_time_ms, windows, args.kline_period)
                eval_futures.append(future)

            eval_results = []
            for future in as_completed(eval_futures):
                try:
                    result = future.result(timeout=600)  # 10 min timeout per individual
                    eval_results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    eval_results.append({
                        "genome": {},
                        "gate_annual": False,
                        "pnl_365d": 0.0,
                        "maxdd_365d": 100.0,
                        "sharpe_365d": 0.0,
                        "pf": None,
                        "trades_365d": 0,
                        "win_rate": 0.0,
                        "fitness": -1000.0,
                        "error": str(e)[:500],
                        "genome_hash": "error",
                    })

        # 2026-09-27 重大修正：as_completed 回傳的是「完成順序」，不是「提交順序」。
        # 先前直接 zip(population, eval_results) 等於把每個基因組配上別人的 fitness：
        #   - 精英保留保留到的是「被誤判為高分」的基因組，真正的最佳解會被丟掉
        #   - 交叉/突變的父代選擇全部基於錯配的 fitness
        #   - best_genome / best_code 存檔的是與該 fitness 無關的基因組
        #   實測症狀：gen 3 的最佳個體在 gen 4 之後就從族群消失，族群品質持續下滑。
        # 改為以 genome_hash 配對，與順序無關。
        _by_hash: Dict[str, Dict[str, Any]] = {}
        for _r in eval_results:
            _h = _r.get("genome_hash")
            if _h is not None and _h not in _by_hash:
                _by_hash[_h] = _r
        for _r in eval_results:
            # Log individual（result 自帶 genome 與 genome_hash，日誌本身自洽）
            all_logs.append({"type": "individual", "generation": gen, **_r})

        # 按 population 順序配對正確的 result
        pop_fitness = []
        eval_by_ind = []
        for ind in population:
            result = _by_hash.get(ind.hash())
            if result is None:
                result = {
                    "genome": ind.to_dict(), "gate_annual": False, "pnl_365d": 0.0,
                    "maxdd_365d": 100.0, "sharpe_365d": 0.0, "pf": None,
                    "trades_365d": 0, "win_rate": 0.0, "fitness": -1000.0,
                    "genome_hash": ind.hash(), "error": "missing evaluation result",
                }
            fitness = result.get("fitness", -1000.0)
            pop_fitness.append((ind, fitness))
            eval_by_ind.append(result)

        # Sort by fitness
        pop_fitness.sort(key=lambda x: x[1], reverse=True)
        # eval_by_ind 目前仍是 population 的原始順序（迴圈是 for ind in population）。
        # pop_fitness 排序後，兩者索引已經對不上 —— eval_by_ind[0] 是 population[0]
        # 的結果，不是最佳個體。2026-09-29 實測：Gen 45 印出 pnl=10.93%，但真正的
        # best 是 pnl=15.20%（fitness 0.0164 vs -0.0857，排名第 3 的個體被誤報為 best）。
        # 一併依同一順序排序，讓 eval_by_ind[i] 對應 pop_fitness[i]。
        eval_by_ind.sort(key=lambda r: r.get("fitness", -1000.0), reverse=True)

        # Generation summary
        gen_best = pop_fitness[0]
        gen_worst = pop_fitness[-1]
        gen_avg = sum(f for _, f in pop_fitness) / len(pop_fitness)
        div = population_diversity(population)
        passed = sum(1 for r in eval_by_ind if r.get("gate_annual", False))

        logger.info(f"Gen {gen}: best={gen_best[1]:.3f} avg={gen_avg:.3f} worst={gen_worst[1]:.3f} div={div:.4f} passed={passed}/{args.pop_size}")
        logger.info(f"  Best: pnl={eval_by_ind[0].get('pnl_365d', 0):.1f}% dd={eval_by_ind[0].get('maxdd_365d', 100):.1f}% sharpe={eval_by_ind[0].get('sharpe_365d', 0):.3f} pf={eval_by_ind[0].get('pf')}")

        # Check improvement
        if gen_best[1] > best_fitness:
            best_fitness = gen_best[1]
            best_genome = deepcopy(gen_best[0])
            best_gen = gen
            no_improve_count = 0
            # Write best code immediately
            try:
                best_code_path.write_text(render_strategy(best_genome, kline_period=args.kline_period), encoding="utf-8")
            except Exception:
                pass
            # Store best individual metrics
            # gen_best 來自排序後的 pop_fitness[0]，eval_by_ind[0] 現在與它對齊
            best_genome_pnl = eval_by_ind[0].get("pnl_365d", 0)
            best_genome_maxdd = eval_by_ind[0].get("maxdd_365d", 100)
            best_genome_sharpe = eval_by_ind[0].get("sharpe_365d", 0)
            # Log best overall
            all_logs.append({
                "type": "best_overall",
                "generation": gen,
                "genome": best_genome.to_dict(),
                "fitness": best_fitness,
                "details": eval_by_ind[0],
            })
        else:
            no_improve_count += 1

        # Generation summary log
        all_logs.append({
            "type": "generation_summary",
            "generation": gen,
            "best_fitness": gen_best[1],
            "avg_fitness": gen_avg,
            "worst_fitness": gen_worst[1],
            "diversity": div,
            "passed": passed,
            "pop_size": args.pop_size,
            "elapsed_sec": time.time() - gen_start,
        })

        # Write logs incrementally
        try:
            with log_file.open("a", encoding="utf-8") as f:
                for rec in all_logs:
                    if "written" not in rec:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        rec["written"] = True
        except Exception as e:
            logger.error(f"failed to write log: {e}")

        # Early stop
        # early_stop <= 0 means "never stop early" — run every generation.
        # The raw comparison `no_improve_count >= args.early_stop` would stop
        # immediately at generation 1 when early_stop == 0, because the
        # improvement counter is 0 and 0 >= 0 is true.
        if args.early_stop > 0 and no_improve_count >= args.early_stop:
            logger.info(f"Early stop: no improvement for {args.early_stop} generations")
            break

        # Create next generation
        if gen < total_generations:
            # Elite preservation
            new_population = [deepcopy(pop_fitness[i][0]) for i in range(min(args.elite, len(pop_fitness)))]

            # Generate offspring
            while len(new_population) < args.pop_size:
                if random.random() < args.cx_rate and len(pop_fitness) >= 2:
                    # Crossover
                    parent1 = tournament_select(pop_fitness, args.tournament)
                    parent2 = tournament_select(pop_fitness, args.tournament)
                    child = crossover(parent1, parent2)
                else:
                    # Clone + mutate
                    parent = tournament_select(pop_fitness, args.tournament)
                    child = deepcopy(parent)
                
                # Mutation - use aggressive mode if fitness stagnating
                stagnant_gens = gen - best_gen if best_gen > 0 else 0
                use_aggressive = stagnant_gens >= 3  # Aggressive after 3 gens no improvement
                child = mutate(child, args.mut_rate, aggressive=use_aggressive)
                new_population.append(child)

            population = new_population

    # Final summary
    summary = {
        "generations": gen,
        "pop_size": args.pop_size,
        "best_fitness": best_fitness,
        "best_generation": best_gen,
        "best_genome": best_genome.to_dict() if best_genome else None,
        "best_code_path": str(best_code_path),
        "log_file": str(log_file),
        "exchange": args.exchange,
        "backtest_days": args.backtest_days,
        "period_years": max(args.backtest_days / 365.0, 1.0),
        "symbols": symbols,
        "early_stopped": no_improve_count >= args.early_stop,
    }

    try:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"failed to write summary: {e}")

    logger.info(f"=== GA Annual complete ===")
    logger.info(f"generations: {gen}, best fitness: {best_fitness:.3f} at gen {best_gen}")
    if best_genome:
        logger.info(f"best genome: {best_genome.to_dict()}")

    return {
        "best_fitness": best_fitness,
        "best_genome": best_genome.to_dict() if best_genome else None,
        "best_pnl_365d": best_genome_pnl if best_genome else None,
        "best_maxdd_365d": best_genome_maxdd if best_genome else None,
        "best_sharpe_365d": best_genome_sharpe if best_genome else None,
        "generations": gen,
        "best_generation": best_gen,
        "log_file": str(log_file),
        "best_code_path": str(best_code_path),
        "summary_path": str(summary_path),
    }


def run_ga_iteration(
    generations: int = 25,
    pop_size: int = 20,
    workers: int = 4,
    elite: int = 2,
    tournament: int = 3,
    cx_rate: float = 0.7,
    mut_rate: float = 0.2,
    early_stop: int = 8,
    min_days: int = 365,
    target_pnl_pct: float = 120.0,
    max_maxdd_pct: float = 5.0,
    resume: bool = False,
    seed: int = None,
    exchange: str = "hyperliquid",
    symbols: List[str] = None,
    skip_validation: bool = False,
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    GA 年度迭代主入口函數，供外部調用
    
    Args:
        generations: GA 世代數
        pop_size: 族群大小
        workers: 平行 workers
        elite: 精英保留數
        tournament: 錦標賽大小
        cx_rate: 交配率
        mut_rate: 突變率
        early_stop: 早停代數
        min_days: 最少回測天數 (強制 365+)
        target_pnl_pct: 目標年化報酬%
        max_maxdd_pct: 最大回撤%上限
        resume: 是否續跑
        seed: 隨機種子
        exchange: 交易所
        symbols: 幣種列表
        skip_validation: 跳過驗證
        output_dir: 輸出目錄
    
    Returns:
        Dict with iteration results
    """
    import sys
    sys.argv = [
        "iterate_annual_ga.py",
        f"--generations={generations}",
        f"--pop-size={pop_size}",
        f"--workers={workers}",
        f"--elite={elite}",
        f"--tournament={tournament}",
        f"--cx-rate={cx_rate}",
        f"--mut-rate={mut_rate}",
        f"--early-stop={early_stop}",
        f"--exchange={exchange}",
    ]
    if symbols:
        sys.argv.append(f"--symbols={','.join(symbols)}")
    if resume:
        sys.argv.append("--resume")
    if seed is not None:
        sys.argv.append(f"--seed={seed}")
    
    return main()


if __name__ == "__main__":
    main()

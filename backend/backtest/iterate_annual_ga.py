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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

# Add backend to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backtest.ga_annual import (
    Genome, random_genome, clamp, calculate_fitness,
    tournament_select, crossover, mutate, population_diversity,
    render_strategy, evaluate_individual, BOUNDS
)

logger = logging.getLogger(__name__)


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
    parser.add_argument("--early-stop", type=int, default=5, help="early stop if no improvement for N generations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
        # Generation 0: random + baseline
        baseline = Genome()  # default baseline
        population = [baseline]
        for _ in range(args.pop_size - 1):
            population.append(random_genome())
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

    log_file.parent.mkdir(parents=True, exist_ok=True)

    total_generations = args.generations
    start_gen = completed_generations + 1

    for gen in range(start_gen, total_generations + 1):
        gen_start = time.time()
        logger.info(f"=== Generation {gen}/{total_generations} ===")

        # Evaluate population in parallel
        eval_futures = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for ind in population:
                future = executor.submit(evaluate_individual, ind.to_dict(), symbols, args.exchange)
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

        # Pair genomes with fitness
        pop_fitness = []
        for ind, result in zip(population, eval_results):
            fitness = result.get("fitness", -1000.0)
            pop_fitness.append((ind, fitness))
            # Log individual
            log_rec = {"type": "individual", "generation": gen, **result}
            all_logs.append(log_rec)

        # Sort by fitness
        pop_fitness.sort(key=lambda x: x[1], reverse=True)

        # Generation summary
        gen_best = pop_fitness[0]
        gen_worst = pop_fitness[-1]
        gen_avg = sum(f for _, f in pop_fitness) / len(pop_fitness)
        div = population_diversity(population)
        passed = sum(1 for _, r in zip(population, eval_results) if r.get("gate_annual", False))

        logger.info(f"Gen {gen}: best={gen_best[1]:.3f} avg={gen_avg:.3f} worst={gen_worst[1]:.3f} div={div:.4f} passed={passed}/{args.pop_size}")
        logger.info(f"  Best: pnl={eval_results[0].get('pnl_365d', 0):.1f}% dd={eval_results[0].get('maxdd_365d', 100):.1f}% sharpe={eval_results[0].get('sharpe_365d', 0):.3f} pf={eval_results[0].get('pf')}")

        # Check improvement
        if gen_best[1] > best_fitness:
            best_fitness = gen_best[1]
            best_genome = deepcopy(gen_best[0])
            best_gen = gen
            no_improve_count = 0
            # Write best code immediately
            try:
                best_code_path.write_text(render_strategy(best_genome), encoding="utf-8")
            except Exception:
                pass
            # Log best overall
            all_logs.append({
                "type": "best_overall",
                "generation": gen,
                "genome": best_genome.to_dict(),
                "fitness": best_fitness,
                "details": eval_results[0],
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
        if no_improve_count >= args.early_stop:
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
                
                # Mutation
                child = mutate(child, args.mut_rate)
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


if __name__ == "__main__":
    main()

"""
Testnet dry-run trigger loop.

Fires program_execution_service._execute_scheduled_trigger for a given
binding every trigger_interval seconds, logging one line per run.
Ctrl-C to stop. Respects _should_trigger_scheduled via last_trigger_at
reset so repeated 60s fires are treated as legitimate triggers.

Usage:
    cd backend
    set -a && . ../.env && set +a
    .venv/bin/python scripts/run_testnet_loop.py [--binding 1] [--symbol BTC] [--max-runs 10]
"""
import argparse
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")

from services.program_execution_service import program_execution_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=int, default=1)
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--max-runs", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    program_execution_service._refresh_binding_cache()
    if args.binding not in program_execution_service._binding_states:
        print(f"Binding {args.binding} not in cache: {list(program_execution_service._binding_states)}")
        sys.exit(1)

    interval = program_execution_service._binding_states[args.binding]["trigger_interval"]
    print(f"Loop: binding={args.binding} symbol={args.symbol} interval={interval}s (Ctrl-C to stop)")

    runs = 0
    ok = 0
    fail = 0
    try:
        while True:
            event_time = datetime.now(timezone.utc)
            # Pretend the last trigger was a full interval ago so
            # _should_trigger_scheduled passes on every iteration.
            program_execution_service._binding_states[args.binding]["last_trigger_at"] = (
                event_time - timedelta(seconds=interval + 5)
            )
            try:
                program_execution_service._execute_scheduled_trigger(
                    args.binding, args.symbol, event_time
                )
                ok += 1
                print(f"[{event_time:%H:%M:%S}] run={runs + 1} fired ok_total={ok}")
            except Exception as e:  # keep the loop alive; errors are in DB logs
                fail += 1
                print(f"[{event_time:%H:%M:%S}] run={runs + 1} ERROR fail_total={fail}: {type(e).__name__}: {e}")

            runs += 1
            if args.max_runs and runs >= args.max_runs:
                print(f"Reached max-runs={args.max_runs}. ok={ok} fail={fail}")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nStopped. runs={runs} ok={ok} fail={fail}")


if __name__ == "__main__":
    main()

# Backend runtime state (manual, survives compaction)
- backend: uvicorn main:app port 8802, health healthy v0.9.12
- start cmd: cd backend && set -a; . ../.env; set +a; .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8802 | tee /tmp/hyper-arena-backend.log
- MUST load .env (HYPERLIQUID_ENCRYPTION_KEY) or snapshot/live decrypt fails
- strategy: trigger_interval 900 (15min), program 2 v3-longonly-testnet-safe-24 (hyperliquid testnet, removed ZEC halted + XRP/ENA/UNI/LINK/CRV missing, base r498 sharpe1.248), program 1 demo binance
- watchlist hyperliquid 10: BTC,ETH,SOL,HYPE,DOGE,SUI,ARB,AVAX,BNB,NEAR (MAX 10)
- live 2026-09-09: program2 testnet-safe-24 live, ZORA Long 44137 entry 0.008796 uPnL -7.33, equity 949.35, avail 758.9; recent fills SKR/ZORA/ADA/PUMP success, no halted since ZEC removal; realistic epoch3 running 75/90 best_any r27 0.413
- iteration realistic: epoch5 running (150 rounds, resume, threshold 0.35/0.10/15, 10 coins BTC/ETH/SOL/HYPE/DOGE/SUI/ARB/AVAX/BNB/NEAR); best_any 0.413 round27 (completed>=120/150, passed 0); baseline best_code_realistic.py tracked in git, candidate identical so ignored; Top30 best_any 1.248 round498 (30 coins), Top30 relaxed 30/30 sharpe -0.69 (baseline drift, keep global best)
- cold-data pcloud symlinks: artifacts in backend/backtest -> ~/pcloud-drive/cold-data/Hyper-Alpha-Arena/backtest/ (iteration_log*.jsonl, summary.json/top30, best_code_top30*, iterate.py.bak*, round_codes/ per-round dump, v2/v3/v4_longonly.py, best_code_old_top30.py, iterate_run_521_530.log); tracked in git: best_code_realistic.py + summary_realistic.json; ignored: *_candidate.py, iteration_log*.jsonl, round_codes, *.bak*

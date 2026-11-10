"""Champion ledger：策略迭代的長期記憶。

策略不會永遠有效。這個模組只做一件事：把每一任 champion 的基因組、
上任時的績效、以及後續重新評估的績效記錄下來，讓「衰退」變成可量測的數字，
而不是等到實盤虧損才發現。

檔案格式（JSONL，append-only，不覆寫歷史）：

    {"event": "promote", "ts": ..., "genome": {...}, "period": "1h",
     "window_days": 180, "end_time_ms": ..., "metrics": {...}, "fitness": ...,
     "run_id": "...", "note": "..."}

    {"event": "reeval", "ts": ..., "run_id": "<被重新評估的那筆>",
     "genome_id": "<sha256 前 16>", "window_days": 90, "end_time_ms": ...,
     "metrics": {...}, "fitness": ..., "decay_pnl": ..., "decay_sharpe": ...}

用法：
    from backtest.champion_ledger import ChampionLedger
    led = ChampionLedger(Path("/tmp/champion_ledger.jsonl"))
    led.promote(genome, metrics, fitness, period="2h", window_days=365, ...)
    rows = led.load()
    best = led.current()          # 最新一任 champion
    led.reeval(best, metrics2, decay=...)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def genome_id(genome: Dict[str, Any]) -> str:
    """基因組的穩定識別碼（排序後的 key 序列化後取 sha256 前 16 碼）。"""
    payload = json.dumps(genome, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ChampionLedger:
    """append-only 的 champion 記錄。"""

    def __init__(self, path: Path):
        self.path = Path(path)

    # ---------- 寫入 ----------

    def _append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def promote(
        self,
        genome: Dict[str, Any],
        metrics: Dict[str, Any],
        fitness: float,
        period: str,
        window_days: int,
        end_time_ms: int,
        run_id: Optional[str] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        """記錄一任新 champion。"""
        return self._append({
            "event": "promote",
            "ts": int(time.time() * 1000),
            "genome": genome,
            "genome_id": genome_id(genome),
            "period": period,
            "window_days": int(window_days),
            "end_time_ms": int(end_time_ms),
            "metrics": metrics,
            "fitness": float(fitness),
            "run_id": run_id or "",
            "note": note,
        })

    def reeval(
        self,
        champion: Dict[str, Any],
        metrics: Dict[str, Any],
        fitness: float,
        window_days: int,
        end_time_ms: int,
        note: str = "",
    ) -> Dict[str, Any]:
        """重新評估某任 champion（通常是更近的視窗），量測衰退。

        decay_pnl / decay_sharpe 是「新值 - 上任值」的絕對差。
        負數代表退步。基準取自該 champion 的 promote 記錄。
        """
        base = champion.get("metrics") or {}
        cur_pnl = float(metrics.get("pnl_percent") or metrics.get("pnl") or 0.0)
        base_pnl = float(base.get("pnl_percent") or base.get("pnl") or 0.0)
        cur_sharpe = float(metrics.get("sharpe") or 0.0)
        base_sharpe = float(base.get("sharpe") or 0.0)
        return self._append({
            "event": "reeval",
            "ts": int(time.time() * 1000),
            # kind 必須繼承，否則 spec 記錄的 reeval 會被分派誤判為 genome。
            "kind": champion.get("kind") or "genome",
            "genome_id": champion.get("genome_id") or genome_id(champion.get("genome") or champion.get("spec") or {}),
            "period": champion.get("period"),
            "window_days": int(window_days),
            "end_time_ms": int(end_time_ms),
            "metrics": metrics,
            "fitness": float(fitness),
            "decay_pnl": cur_pnl - base_pnl,
            "decay_sharpe": cur_sharpe - base_sharpe,
            "promoted_at": champion.get("ts"),
            "note": note,
        })

    # ---------- 讀取 ----------

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def promotions(self) -> List[Dict[str, Any]]:
        return [r for r in self.load() if r.get("event") == "promote"]

    def current(self) -> Optional[Dict[str, Any]]:
        """最新一任 champion（最後一筆 promote）。"""
        promos = self.promotions()
        return promos[-1] if promos else None

    def history_for(self, gid: str) -> List[Dict[str, Any]]:
        return [r for r in self.load() if r.get("genome_id") == gid]

    def decay_report(self) -> List[Dict[str, Any]]:
        """所有 reeval 記錄，依時間排序——這就是「策略會失效」的證據表。"""
        rows = [r for r in self.load() if r.get("event") == "reeval"]
        return sorted(rows, key=lambda r: r.get("ts") or 0)

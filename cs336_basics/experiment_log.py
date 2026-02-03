from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        csv_path: str | Path,
        *,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="")
        self._writer: csv.DictWriter[str] | None = None
        self._use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            import wandb

            self._wandb = wandb
            self._wandb.init(project=wandb_project or "cs336-basics", name=run_name, config=config)

    def log(self, step: int, **metrics: float) -> None:
        row = {"step": step, **metrics}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
        self._file.close()

"""Episode logger — writes JSONL transcripts and CSV reward/curriculum logs."""

from __future__ import annotations

import csv
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from models import RewardBreakdown, TrialAction, TrialObservation
from server.config import settings

logger = logging.getLogger(__name__)


class EpisodeLogger:
    """Logs per-step JSONL records and end-of-episode CSV rows.

    All file I/O is wrapped in try/except — failures emit a warning and
    never propagate to the caller (Req 7.3).
    """

    def __init__(
        self,
        log_path: Path | None = None,
        episode_id: str | None = None,
        curriculum_tier: int = 0,
    ) -> None:
        self._log_path: Path = log_path if log_path is not None else settings.log_path
        self._episode_id: str = (
            episode_id if episode_id is not None else str(uuid.uuid4())
        )
        self._curriculum_tier: int = curriculum_tier

        # Derived paths
        self._transcripts_dir: Path = self._log_path / "episode_transcripts"
        self._transcript_file: Path = (
            self._transcripts_dir / f"{self._episode_id}.jsonl"
        )
        self._reward_csv: Path = self._log_path / "reward_log.csv"
        self._curriculum_csv: Path = self._log_path / "curriculum_log.csv"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def log_step(
        self,
        step_idx: int,
        action: TrialAction,
        obs: TrialObservation,
        reward: RewardBreakdown,
        done: bool,
    ) -> None:
        """Append one JSONL record for this step to the episode transcript (Req 7.1)."""
        record = {
            "step_index": step_idx,
            "action": action.model_dump(),
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
        }
        self._append_jsonl(record)

    def log_summary(
        self,
        scenario_id: str,
        total_reward: float,
        episode_length: int,
        terminal_outcome: str,
    ) -> None:
        """Write summary JSONL record and append rows to both CSV files (Req 7.2)."""
        summary = {
            "type": "summary",
            "episode_id": self._episode_id,
            "scenario_id": scenario_id,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "terminal_outcome": terminal_outcome,
        }
        self._append_jsonl(summary)
        self._append_reward_csv(
            scenario_id, total_reward, episode_length, terminal_outcome
        )
        self._append_curriculum_csv(scenario_id, terminal_outcome)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_transcripts_dir(self) -> bool:
        """Create the transcripts directory if needed. Returns True on success."""
        try:
            self._transcripts_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:
            logger.warning(
                "EpisodeLogger: cannot create transcripts directory %s: %s",
                self._transcripts_dir,
                exc,
            )
            return False

    def _append_jsonl(self, record: dict) -> None:
        """Append a JSON line to the episode transcript file."""
        if not self._ensure_transcripts_dir():
            return
        try:
            with self._transcript_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.warning(
                "EpisodeLogger: cannot write to transcript %s: %s",
                self._transcript_file,
                exc,
            )

    def _append_reward_csv(
        self,
        scenario_id: str,
        total_reward: float,
        episode_length: int,
        terminal_outcome: str,
    ) -> None:
        """Append a row to reward_log.csv, writing the header if the file is new."""
        _REWARD_HEADERS = [
            "episode_id",
            "scenario_id",
            "total_reward",
            "episode_length",
            "terminal_outcome",
            "curriculum_tier",
        ]
        try:
            self._log_path.mkdir(parents=True, exist_ok=True)
            write_header = not self._reward_csv.exists()
            with self._reward_csv.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(_REWARD_HEADERS)
                writer.writerow(
                    [
                        self._episode_id,
                        scenario_id,
                        total_reward,
                        episode_length,
                        terminal_outcome,
                        self._curriculum_tier,
                    ]
                )
        except OSError as exc:
            logger.warning(
                "EpisodeLogger: cannot write to reward CSV %s: %s",
                self._reward_csv,
                exc,
            )

    def _append_curriculum_csv(self, scenario_id: str, terminal_outcome: str) -> None:
        """Append a row to curriculum_log.csv, writing the header if the file is new."""
        _CURRICULUM_HEADERS = [
            "episode_id",
            "scenario_id",
            "curriculum_tier",
            "terminal_outcome",
        ]
        try:
            self._log_path.mkdir(parents=True, exist_ok=True)
            write_header = not self._curriculum_csv.exists()
            with self._curriculum_csv.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(_CURRICULUM_HEADERS)
                writer.writerow(
                    [
                        self._episode_id,
                        scenario_id,
                        self._curriculum_tier,
                        terminal_outcome,
                    ]
                )
        except OSError as exc:
            logger.warning(
                "EpisodeLogger: cannot write to curriculum CSV %s: %s",
                self._curriculum_csv,
                exc,
            )

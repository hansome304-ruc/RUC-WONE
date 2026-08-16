from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medicine_agentic.models import SkillName


DEFAULT_MAX_ATTEMPTS = {
    SkillName.PICK_CARTON.value: 2,
    SkillName.PLACE_CARTON.value: 2,
    SkillName.STABILIZE_CARTON.value: 2,
    SkillName.PICK_ITEM.value: 2,
    SkillName.INSERT_ITEM.value: 2,
    SkillName.ERECT_CARTON.value: 2,
    SkillName.CLOSE_CARTON.value: 2,
    SkillName.SAFE_MOVE.value: 1,
    SkillName.VERIFY.value: 1,
}


@dataclass(frozen=True)
class MedicineConfig:
    blister_count: int = 3
    pack_slots: tuple[str, ...] = ("R01",)
    pack_orientation: str = "LABEL_FORWARD"
    max_attempts: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MAX_ATTEMPTS)
    )

    def __post_init__(self) -> None:
        if self.blister_count <= 0:
            raise ValueError("blister_count must be positive")
        if not self.pack_slots:
            raise ValueError("pack_slots must not be empty")
        invalid = {
            name: attempts
            for name, attempts in self.max_attempts.items()
            if int(attempts) <= 0
        }
        if invalid:
            raise ValueError(f"max_attempts values must be positive: {invalid}")

    def attempts_for(self, skill: SkillName) -> int:
        return int(self.max_attempts.get(skill.value, 1))

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "MedicineConfig":
        attempts = dict(DEFAULT_MAX_ATTEMPTS)
        attempts.update(
            {str(name): int(value) for name, value in raw.get("max_attempts", {}).items()}
        )
        return cls(
            blister_count=int(raw.get("blister_count", 3)),
            pack_slots=tuple(str(slot) for slot in raw.get("pack_slots", ("R01",))),
            pack_orientation=str(raw.get("pack_orientation", "LABEL_FORWARD")),
            max_attempts=attempts,
        )


def load_config(path: str | Path) -> MedicineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be an object: {config_path}")
    return MedicineConfig.from_mapping(raw)


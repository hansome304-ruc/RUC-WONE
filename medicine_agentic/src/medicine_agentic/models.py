from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowName(str, Enum):
    PACK = "pack"
    LOAD = "load"
    ERECT = "erect"


class SkillName(str, Enum):
    PICK_CARTON = "pick_carton"
    PLACE_CARTON = "place_carton"
    STABILIZE_CARTON = "stabilize_carton"
    PICK_ITEM = "pick_item"
    INSERT_ITEM = "insert_item"
    ERECT_CARTON = "erect_carton"
    CLOSE_CARTON = "close_carton"
    SAFE_MOVE = "safe_move"
    VERIFY = "verify"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class SkillRequest:
    skill: SkillName
    params: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1


@dataclass(frozen=True)
class SkillResult:
    ok: bool
    retryable: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(cls, **evidence: Any) -> "SkillResult":
        return cls(ok=True, evidence=evidence)

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        retryable: bool,
        **evidence: Any,
    ) -> "SkillResult":
        return cls(
            ok=False,
            retryable=retryable,
            evidence=evidence,
            error=error,
        )


@dataclass(frozen=True)
class WorkflowEvent:
    sequence: int
    workflow: WorkflowName
    state: str
    request: SkillRequest
    result: SkillResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "workflow": self.workflow.value,
            "state": self.state,
            "skill": self.request.skill.value,
            "attempt": self.request.attempt,
            "params": dict(self.request.params),
            "ok": self.result.ok,
            "retryable": self.result.retryable,
            "evidence": dict(self.result.evidence),
            "error": self.result.error,
        }


@dataclass(frozen=True)
class WorkflowReport:
    workflow: WorkflowName
    ok: bool
    completed_units: int
    events: tuple[WorkflowEvent, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.value,
            "ok": self.ok,
            "completed_units": self.completed_units,
            "error": self.error,
            "events": [event.to_dict() for event in self.events],
        }


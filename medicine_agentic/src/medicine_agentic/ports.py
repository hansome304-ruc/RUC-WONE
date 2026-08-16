from __future__ import annotations

from typing import Protocol

from medicine_agentic.models import SkillRequest, SkillResult


class SkillExecutor(Protocol):
    """Boundary implemented by dry-run, hardware and learned-policy adapters."""

    def execute(self, request: SkillRequest) -> SkillResult: ...

    def safe_stop(self, reason: str) -> None: ...


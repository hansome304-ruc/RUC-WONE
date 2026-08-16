from __future__ import annotations

from collections.abc import Callable

from medicine_agentic.config import MedicineConfig
from medicine_agentic.models import (
    SkillName,
    SkillRequest,
    SkillResult,
    WorkflowEvent,
    WorkflowName,
    WorkflowReport,
)
from medicine_agentic.ports import SkillExecutor


class WorkflowFailure(RuntimeError):
    pass


class MedicineWorkflow:
    """Deterministic orchestration; learned policies live behind SkillExecutor."""

    def __init__(self, config: MedicineConfig, executor: SkillExecutor) -> None:
        self.config = config
        self.executor = executor
        self._events: list[WorkflowEvent] = []
        self._sequence = 0

    def run_pack(self) -> WorkflowReport:
        return self._run_guarded(WorkflowName.PACK, self._run_pack_steps)

    def run_load(self) -> WorkflowReport:
        return self._run_guarded(WorkflowName.LOAD, self._run_load_steps)

    def run_erect(self) -> WorkflowReport:
        return self._run_guarded(WorkflowName.ERECT, self._run_erect_steps)

    def _run_guarded(
        self,
        workflow: WorkflowName,
        steps: Callable[[WorkflowName], int],
    ) -> WorkflowReport:
        self._events = []
        self._sequence = 0
        completed_units = 0
        error: str | None = None
        try:
            completed_units = steps(workflow)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                self.executor.safe_stop(error)
                self._record(
                    workflow,
                    "failed",
                    SkillRequest(SkillName.SAFE_STOP, {"reason": error}),
                    SkillResult.success(reason=error),
                )
            except Exception as stop_exc:
                error = f"{error}; safe_stop failed: {type(stop_exc).__name__}: {stop_exc}"
        return WorkflowReport(
            workflow=workflow,
            ok=error is None,
            completed_units=completed_units,
            events=tuple(self._events),
            error=error,
        )

    def _run_pack_steps(self, workflow: WorkflowName) -> int:
        completed = 0
        self._execute(workflow, "pack_pick_ready", SkillName.SAFE_MOVE, target="pack_pick_ready")
        for index, slot_id in enumerate(self.config.pack_slots, start=1):
            self._execute(
                workflow,
                f"pick_carton_{index}",
                SkillName.PICK_CARTON,
                mode="CLOSED_CARTON",
            )
            self._execute(
                workflow,
                f"verify_carton_held_{index}",
                SkillName.VERIFY,
                target="carton_held",
                mode="CLOSED_CARTON",
            )
            self._execute(
                workflow,
                f"move_to_slot_{slot_id}",
                SkillName.SAFE_MOVE,
                target="pack_preinsert",
                slot_id=slot_id,
            )
            self._execute(
                workflow,
                f"place_slot_{slot_id}",
                SkillName.PLACE_CARTON,
                slot_id=slot_id,
                orientation=self.config.pack_orientation,
            )
            self._execute(
                workflow,
                f"verify_slot_{slot_id}",
                SkillName.VERIFY,
                target="slot_occupied",
                slot_id=slot_id,
                orientation=self.config.pack_orientation,
            )
            completed += 1
        self._execute(workflow, "pack_home", SkillName.SAFE_MOVE, target="home")
        return completed

    def _run_load_steps(self, workflow: WorkflowName) -> int:
        completed = 0
        self._execute(
            workflow,
            "stabilize_carton",
            SkillName.STABILIZE_CARTON,
            mode="OPEN_CARTON",
        )
        for stage in range(1, self.config.blister_count + 1):
            self._execute(
                workflow,
                f"pick_blister_{stage}",
                SkillName.PICK_ITEM,
                item_type="BLISTER",
                stage=stage,
            )
            self._execute(
                workflow,
                f"verify_blister_held_{stage}",
                SkillName.VERIFY,
                target="single_item_held",
                item_type="BLISTER",
                stage=stage,
            )
            self._execute(
                workflow,
                f"move_blister_to_preinsert_{stage}",
                SkillName.SAFE_MOVE,
                target="blister_preinsert",
                stage=stage,
            )
            self._execute(
                workflow,
                f"insert_blister_{stage}",
                SkillName.INSERT_ITEM,
                item_type="BLISTER",
                stage=stage,
            )
            self._execute(
                workflow,
                f"verify_blister_inserted_{stage}",
                SkillName.VERIFY,
                target="item_inserted",
                item_type="BLISTER",
                stage=stage,
            )
            completed += 1

        self._execute(
            workflow,
            "pick_leaflet",
            SkillName.PICK_ITEM,
            item_type="LEAFLET",
            stage="LEAFLET",
        )
        self._execute(
            workflow,
            "verify_leaflet_held",
            SkillName.VERIFY,
            target="single_item_held",
            item_type="LEAFLET",
            stage="LEAFLET",
        )
        self._execute(
            workflow,
            "move_leaflet_to_preinsert",
            SkillName.SAFE_MOVE,
            target="leaflet_preinsert",
        )
        self._execute(
            workflow,
            "insert_leaflet",
            SkillName.INSERT_ITEM,
            item_type="LEAFLET",
            stage="LEAFLET",
        )
        self._execute(
            workflow,
            "verify_leaflet_inserted",
            SkillName.VERIFY,
            target="item_inserted",
            item_type="LEAFLET",
            stage="LEAFLET",
        )
        completed += 1
        self._execute(
            workflow,
            "close_loaded_carton",
            SkillName.CLOSE_CARTON,
            mode="LOADED_TOP",
        )
        self._execute(
            workflow,
            "verify_loaded_carton_closed",
            SkillName.VERIFY,
            target="carton_closed",
            mode="LOADED_TOP",
        )
        self._execute(workflow, "load_home", SkillName.SAFE_MOVE, target="home")
        return completed

    def _run_erect_steps(self, workflow: WorkflowName) -> int:
        self._execute(
            workflow,
            "pick_flat_blank",
            SkillName.PICK_CARTON,
            mode="FLAT_BLANK",
        )
        self._execute(
            workflow,
            "verify_flat_blank_held",
            SkillName.VERIFY,
            target="carton_held",
            mode="FLAT_BLANK",
        )
        self._execute(workflow, "erect_carton", SkillName.ERECT_CARTON)
        self._execute(
            workflow,
            "verify_carton_squared",
            SkillName.VERIFY,
            target="carton_squared",
        )
        self._execute(
            workflow,
            "close_empty_bottom",
            SkillName.CLOSE_CARTON,
            mode="EMPTY_BOTTOM",
        )
        self._execute(
            workflow,
            "verify_empty_bottom_closed",
            SkillName.VERIFY,
            target="carton_closed",
            mode="EMPTY_BOTTOM",
        )
        self._execute(workflow, "erect_home", SkillName.SAFE_MOVE, target="home")
        return 1

    def _execute(
        self,
        workflow: WorkflowName,
        state: str,
        skill: SkillName,
        **params: object,
    ) -> SkillResult:
        max_attempts = self.config.attempts_for(skill)
        last_result: SkillResult | None = None
        for attempt in range(1, max_attempts + 1):
            request = SkillRequest(skill=skill, params=dict(params), attempt=attempt)
            result = self.executor.execute(request)
            self._record(workflow, state, request, result)
            if result.ok:
                return result
            last_result = result
            if not result.retryable or attempt >= max_attempts:
                break
            if skill is not SkillName.SAFE_MOVE:
                recovery_request = SkillRequest(
                    skill=SkillName.SAFE_MOVE,
                    params={"target": "reobserve", "failed_state": state},
                    attempt=1,
                )
                recovery_result = self.executor.execute(recovery_request)
                self._record(
                    workflow,
                    f"recover_{state}",
                    recovery_request,
                    recovery_result,
                )
                if not recovery_result.ok:
                    last_result = recovery_result
                    break

        detail = last_result.error if last_result is not None else "no result"
        raise WorkflowFailure(
            f"{workflow.value}:{state}:{skill.value} failed after "
            f"{max_attempts} attempt(s): {detail}"
        )

    def _record(
        self,
        workflow: WorkflowName,
        state: str,
        request: SkillRequest,
        result: SkillResult,
    ) -> None:
        self._sequence += 1
        self._events.append(
            WorkflowEvent(
                sequence=self._sequence,
                workflow=workflow,
                state=state,
                request=request,
                result=result,
            )
        )


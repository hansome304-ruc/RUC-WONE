from __future__ import annotations

from dataclasses import dataclass, field

from medicine_agentic.models import SkillName, SkillRequest, SkillResult


@dataclass
class DryRunState:
    left_payload: str | None = None
    right_carton_stabilized: bool = False
    occupied_slots: set[str] = field(default_factory=set)
    inserted_blisters: int = 0
    leaflet_inserted: bool = False
    carton_squared: bool = False
    carton_closed: bool = False


class DryRunSkillExecutor:
    """State-aware simulator used before any real robot adapter is enabled."""

    def __init__(self, failures: dict[str, int] | None = None) -> None:
        self.state = DryRunState()
        self.failures = dict(failures or {})
        self.calls: list[SkillRequest] = []
        self.safe_stop_calls: list[str] = []

    def execute(self, request: SkillRequest) -> SkillResult:
        self.calls.append(request)
        forced_key = self._forced_failure_key(request)
        if forced_key is not None:
            self.failures[forced_key] -= 1
            return SkillResult.failure(
                f"forced dry-run failure: {forced_key}",
                retryable=True,
                failure_key=forced_key,
            )

        handlers = {
            SkillName.PICK_CARTON: self._pick_carton,
            SkillName.PLACE_CARTON: self._place_carton,
            SkillName.STABILIZE_CARTON: self._stabilize_carton,
            SkillName.PICK_ITEM: self._pick_item,
            SkillName.INSERT_ITEM: self._insert_item,
            SkillName.ERECT_CARTON: self._erect_carton,
            SkillName.CLOSE_CARTON: self._close_carton,
            SkillName.SAFE_MOVE: self._safe_move,
            SkillName.VERIFY: self._verify,
        }
        handler = handlers.get(request.skill)
        if handler is None:
            return SkillResult.failure(
                f"dry-run has no handler for {request.skill.value}",
                retryable=False,
            )
        return handler(request)

    def safe_stop(self, reason: str) -> None:
        self.safe_stop_calls.append(reason)

    def _pick_carton(self, request: SkillRequest) -> SkillResult:
        if self.state.left_payload is not None:
            return SkillResult.failure(
                f"left arm already holds {self.state.left_payload}",
                retryable=False,
            )
        mode = str(request.params["mode"])
        self.state.left_payload = mode
        self.state.carton_closed = mode == "CLOSED_CARTON"
        self.state.carton_squared = mode != "FLAT_BLANK"
        return SkillResult.success(mode=mode, vacuum_ok=True, pose="carry_ready")

    def _place_carton(self, request: SkillRequest) -> SkillResult:
        if self.state.left_payload != "CLOSED_CARTON":
            return SkillResult.failure("no closed carton is held", retryable=False)
        slot_id = str(request.params["slot_id"])
        self.state.occupied_slots.add(slot_id)
        self.state.left_payload = None
        return SkillResult.success(
            slot_id=slot_id,
            orientation=request.params["orientation"],
            released=True,
        )

    def _stabilize_carton(self, request: SkillRequest) -> SkillResult:
        self.state.right_carton_stabilized = True
        self.state.carton_closed = False
        return SkillResult.success(
            mode=request.params["mode"],
            carton_stable=True,
            mouth_visible=True,
        )

    def _pick_item(self, request: SkillRequest) -> SkillResult:
        if self.state.left_payload is not None:
            return SkillResult.failure(
                f"left arm already holds {self.state.left_payload}",
                retryable=False,
            )
        item_type = str(request.params["item_type"])
        self.state.left_payload = item_type
        return SkillResult.success(
            item_type=item_type,
            stage=request.params.get("stage"),
            vacuum_ok=True,
            single_item=True,
            pose="carry_ready",
        )

    def _insert_item(self, request: SkillRequest) -> SkillResult:
        item_type = str(request.params["item_type"])
        if not self.state.right_carton_stabilized:
            return SkillResult.failure("carton is not stabilized", retryable=False)
        if self.state.left_payload != item_type:
            return SkillResult.failure(
                f"left arm does not hold {item_type}",
                retryable=False,
            )
        if item_type == "BLISTER":
            stage = int(request.params["stage"])
            if stage != self.state.inserted_blisters + 1:
                return SkillResult.failure(
                    f"unexpected blister stage {stage}",
                    retryable=False,
                )
            self.state.inserted_blisters = stage
        elif item_type == "LEAFLET":
            self.state.leaflet_inserted = True
        else:
            return SkillResult.failure(
                f"unknown item_type {item_type}",
                retryable=False,
            )
        self.state.left_payload = None
        return SkillResult.success(
            item_type=item_type,
            stage=request.params.get("stage"),
            inserted=True,
            released=True,
        )

    def _erect_carton(self, request: SkillRequest) -> SkillResult:
        del request
        if self.state.left_payload != "FLAT_BLANK":
            return SkillResult.failure("flat blank is not held", retryable=False)
        self.state.left_payload = "OPEN_CARTON"
        self.state.carton_squared = True
        return SkillResult.success(carton_squared=True, corners_visible=4)

    def _close_carton(self, request: SkillRequest) -> SkillResult:
        mode = str(request.params["mode"])
        if mode == "LOADED_TOP" and not self.state.right_carton_stabilized:
            return SkillResult.failure("loaded carton is not stabilized", retryable=False)
        if mode == "EMPTY_BOTTOM" and self.state.left_payload != "OPEN_CARTON":
            return SkillResult.failure("erected empty carton is not held", retryable=False)
        self.state.carton_closed = True
        if mode == "EMPTY_BOTTOM":
            self.state.left_payload = "CLOSED_CARTON"
        return SkillResult.success(
            mode=mode,
            dust_flaps_inside=True,
            tongue_inserted=True,
            lid_flush=True,
        )

    def _safe_move(self, request: SkillRequest) -> SkillResult:
        return SkillResult.success(target=request.params["target"], collision_free=True)

    def _verify(self, request: SkillRequest) -> SkillResult:
        target = str(request.params["target"])
        checks = {
            "carton_held": self.state.left_payload == request.params.get("mode"),
            "single_item_held": self.state.left_payload == request.params.get("item_type"),
            "slot_occupied": str(request.params.get("slot_id")) in self.state.occupied_slots,
            "carton_squared": self.state.carton_squared,
            "carton_closed": self.state.carton_closed,
        }
        if target == "item_inserted":
            if request.params.get("item_type") == "BLISTER":
                ok = self.state.inserted_blisters >= int(request.params["stage"])
            else:
                ok = self.state.leaflet_inserted
        else:
            ok = checks.get(target, False)
        if not ok:
            return SkillResult.failure(
                f"verification failed: {target}",
                retryable=False,
                target=target,
            )
        return SkillResult.success(target=target, verified=True)

    def _forced_failure_key(self, request: SkillRequest) -> str | None:
        candidates = [self._request_key(request), request.skill.value]
        for key in candidates:
            if self.failures.get(key, 0) > 0:
                return key
        return None

    @staticmethod
    def _request_key(request: SkillRequest) -> str:
        parts = [request.skill.value]
        if "item_type" in request.params:
            parts.append(str(request.params["item_type"]))
        if "stage" in request.params:
            parts.append(str(request.params["stage"]))
        elif "mode" in request.params:
            parts.append(str(request.params["mode"]))
        return ":".join(parts)


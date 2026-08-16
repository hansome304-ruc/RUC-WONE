from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicine_agentic.config import MedicineConfig
from medicine_agentic.dry_run import DryRunSkillExecutor
from medicine_agentic.models import SkillName
from medicine_agentic.workflow import MedicineWorkflow


def test_config() -> MedicineConfig:
    return MedicineConfig(
        blister_count=3,
        pack_slots=("R01", "R02"),
        pack_orientation="LABEL_FORWARD",
    )


class MedicineWorkflowTests(unittest.TestCase):
    def test_all_workflows_succeed(self) -> None:
        executor = DryRunSkillExecutor()
        workflow = MedicineWorkflow(test_config(), executor)

        pack = workflow.run_pack()
        load = workflow.run_load()
        erect = workflow.run_erect()

        self.assertTrue(pack.ok, pack.error)
        self.assertTrue(load.ok, load.error)
        self.assertTrue(erect.ok, erect.error)
        self.assertEqual(pack.completed_units, 2)
        self.assertEqual(load.completed_units, 4)
        self.assertEqual(erect.completed_units, 1)
        self.assertEqual(executor.state.occupied_slots, {"R01", "R02"})
        self.assertEqual(executor.state.inserted_blisters, 3)
        self.assertTrue(executor.state.leaflet_inserted)
        self.assertTrue(executor.state.carton_closed)
        self.assertEqual(executor.safe_stop_calls, [])

    def test_blister_insert_retries_only_current_skill(self) -> None:
        executor = DryRunSkillExecutor(
            failures={"insert_item:BLISTER:2": 1}
        )
        workflow = MedicineWorkflow(test_config(), executor)

        report = workflow.run_load()

        self.assertTrue(report.ok, report.error)
        stage_two_calls = [
            call
            for call in executor.calls
            if call.skill is SkillName.INSERT_ITEM
            and call.params.get("item_type") == "BLISTER"
            and call.params.get("stage") == 2
        ]
        self.assertEqual([call.attempt for call in stage_two_calls], [1, 2])
        self.assertEqual(executor.state.inserted_blisters, 3)

    def test_exhausted_retry_fails_closed(self) -> None:
        executor = DryRunSkillExecutor(
            failures={"pick_carton:CLOSED_CARTON": 2}
        )
        workflow = MedicineWorkflow(test_config(), executor)

        report = workflow.run_pack()

        self.assertFalse(report.ok)
        self.assertIn("pick_carton", report.error or "")
        self.assertEqual(len(executor.safe_stop_calls), 1)
        self.assertEqual(executor.state.occupied_slots, set())


if __name__ == "__main__":
    unittest.main()


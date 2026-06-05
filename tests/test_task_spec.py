from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tcad_agent.task_spec import (
    DeviceKind,
    TaskIntent,
    load_task_spec,
    parse_task_text,
    task_spec_to_loop_request,
    write_task_spec,
)


class TaskSpecTest(unittest.TestCase):
    def test_parse_pn_iv_text_extracts_sweep_and_policies(self) -> None:
        spec = parse_task_text(
            "PN junction IV from 0 to 5 V step 5 V min_step 1.25 V max_attempts 3 max_cycles 4",
            task_id="pn_text",
            use_llm=False,
        )

        self.assertEqual(spec.task_id, "pn_text")
        self.assertEqual(spec.intent, TaskIntent.SIMULATE_IV)
        self.assertEqual(spec.device, DeviceKind.PN_JUNCTION)
        self.assertEqual(spec.sweep.start_v, 0.0)
        self.assertEqual(spec.sweep.stop_v, 5.0)
        self.assertEqual(spec.sweep.step_v, 5.0)
        self.assertEqual(spec.sweep.min_step_v, 1.25)
        self.assertEqual(spec.execution.max_attempts, 3)
        self.assertEqual(spec.execution.max_cycles, 4)
        self.assertFalse(spec.execution.use_llm)

    def test_parse_defaults_to_pn_iv_with_assumptions(self) -> None:
        spec = parse_task_text("扫到 0.5V 步长 0.1V", task_id="defaulted")

        self.assertEqual(spec.sweep.start_v, 0.0)
        self.assertEqual(spec.sweep.stop_v, 0.5)
        self.assertEqual(spec.sweep.step_v, 0.1)
        self.assertGreaterEqual(len(spec.assumptions), 1)

    def test_pn_junction_phrase_does_not_set_junction_position(self) -> None:
        spec = parse_task_text("PN 结 0 到 0.2V IV，步长 0.1V", task_id="pn_phrase")

        self.assertEqual(spec.sweep.start_v, 0.0)
        self.assertEqual(spec.sweep.stop_v, 0.2)
        self.assertEqual(spec.parameters.junction_um, 0.05)

    def test_task_spec_to_loop_request(self) -> None:
        spec = parse_task_text("PN IV 0 to 5 V step 5 V", task_id="loop_req", use_llm=True)
        root = Path("/tmp/actsoft-unit")

        request = task_spec_to_loop_request(
            spec,
            loop_root=root / "loops",
            run_root=root / "agent_tools",
            use_llm=False,
        )

        self.assertEqual(request.loop_id, "loop_req")
        self.assertEqual(request.stop, 5.0)
        self.assertEqual(request.step, 5.0)
        self.assertFalse(request.use_llm)
        self.assertEqual(request.loop_root, root / "loops")

    def test_parse_device_parameters(self) -> None:
        spec = parse_task_text(
            "PN IV 从 0V 扫到 0.5V 步长 0.1V 器件长度 0.2um 结位置 0.08um "
            "p区掺杂 1e17 n区掺杂 2e17 温度 350K 接触网格 0.002um 结网格 0.00002um",
            task_id="param_text",
        )

        self.assertEqual(spec.parameters.length_um, 0.2)
        self.assertEqual(spec.parameters.junction_um, 0.08)
        self.assertEqual(spec.parameters.p_doping_cm3, 1e17)
        self.assertEqual(spec.parameters.n_doping_cm3, 2e17)
        self.assertEqual(spec.parameters.temperature_k, 350.0)
        self.assertEqual(spec.mesh.contact_spacing_um, 0.002)
        self.assertEqual(spec.mesh.junction_spacing_um, 0.00002)

        request = task_spec_to_loop_request(spec)
        self.assertEqual(request.length_um, 0.2)
        self.assertEqual(request.junction_um, 0.08)
        self.assertEqual(request.p_doping_cm3, 1e17)

    def test_write_and_load_task_spec(self) -> None:
        spec = parse_task_text("PN IV to 0.5 V step 0.1 V", task_id="roundtrip")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"

            write_task_spec(spec, path)
            loaded = load_task_spec(path)

        self.assertEqual(loaded.task_id, "roundtrip")
        self.assertEqual(loaded.sweep.stop_v, 0.5)


if __name__ == "__main__":
    unittest.main()

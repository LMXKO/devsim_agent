# Power MOSFET Signoff Gate

`tcad_agent.power_mosfet_signoff` bundles the evidence workflow for the DEVSIM-backed Power MOSFET/LDMOS 2D field-plate runner.

```bash
python3.11 -m tcad_agent.power_mosfet_signoff --run-id ldmos_gate_001
```

It can collect:

- 2D field-plate baseline state;
- physical benchmark;
- tool-level mesh/model convergence over `power_mos_junction_mesh_spacing_um`;
- optional golden/measured curve comparison;
- `signoff_gate.json` with `ready`, `conditional`, or `blocked`.

Without golden/measured correlation or clean convergence evidence, the gate stays conditional/blocked. This is intentional: the runner is useful for autonomous iteration, while final industrial signoff still needs user-owned calibration evidence and process/layout correlation.

# TCAD Task Spec

`TaskSpec` is the handoff format between a user-facing planner and the autonomous TCAD execution loop.

The first supported task is:

```text
PN junction IV sweep with DEVSIM
```

## Text To Task

Create a standardized task from natural language without executing it:

```bash
python3.11 -m tcad_agent.tools.task_runner \
  --task-id pn_iv_text \
  --text "PN junction IV from 0 to 5 V step 5 V min_step 1.25 V max_attempts 3 max_cycles 3" \
  --no-llm
```

This writes:

```text
runs/tasks/pn_iv_text/
  task.json
  task_run_state.json
```

`task_run_state.json` includes the exact `AutonomousLoopRequest` that would be executed.

## Execute A Task

Add `--execute` to run the autonomous loop:

```bash
python3.11 -m tcad_agent.tools.task_runner \
  --task-id pn_iv_text_run \
  --text "PN junction IV from 0 to 5 V step 5 V min_step 1.25 V max_attempts 3 max_cycles 3" \
  --execute \
  --no-llm
```

The task runner records task-level status, while the loop still writes its own checkpoint:

```text
runs/tasks/<task_id>/task_run_state.json
runs/autonomous_loop/<task_id>/loop_state.json
```

## Resume

Resume an existing task by loading its `task.json`:

```bash
python3.11 -m tcad_agent.tools.task_runner \
  --task runs/tasks/pn_iv_text_run/task.json \
  --execute \
  --resume \
  --no-llm
```

## Current Schema

`task.json` contains:

- `schema_version`: currently `actsoft.tcad.task.v1`;
- `task_id`: stable task identifier used as the loop id;
- `intent`: currently `simulate_iv`;
- `device`: currently `pn_junction`;
- `simulator`: currently `devsim`;
- `sweep`: start, stop, step, and minimum retry step;
- `parameters`: PN junction length, junction position, doping, temperature, and carrier lifetimes;
- `mesh`: contact and junction mesh spacing;
- `quality`: deterministic result acceptance policy;
- `execution`: attempts, cycles, timeout, and LLM usage policy;
- `assumptions` and `warnings`: parser decisions that should stay visible to the agent.

The next expansion point is to let an LLM planner produce this same schema for more complex tasks, then keep deterministic validation and execution unchanged.

See [task_planner.md](task_planner.md) for the LLM-assisted planner that produces this schema from free-form text.

## Parameterized Example

```bash
python3.11 -m tcad_agent.tools.task_runner \
  --task-id param_task \
  --text "PN IV 从 0V 扫到 0.2V 步长 0.1V 器件长度 0.2um 结位置 0.08um p区掺杂 1e17 n区掺杂 2e17 温度 325K" \
  --execute \
  --no-llm
```

The generated loop request passes these values down to DEVSIM:

```text
length_um
junction_um
p_doping_cm3
n_doping_cm3
temperature_k
electron_lifetime_s
hole_lifetime_s
contact_spacing_um
junction_spacing_um
```

The runner writes the resolved parameter set into `summary.json` under `parameters`.

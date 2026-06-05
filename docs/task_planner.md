# LLM Task Planner

The LLM task planner converts free-form user text into the same deterministic `TaskSpec` schema used by `task_runner`.

This layer is intentionally advisory:

- the LLM interprets user intent;
- deterministic code repairs aliases and missing fields;
- Pydantic validates the final schema;
- invalid LLM output can fall back to the deterministic parser.

## Plan Only

Use the configured OpenAI-compatible endpoint to produce a task plan:

```bash
python3.11 -m tcad_agent.tools.task_planner \
  --task-id planner_smoke \
  --text "做一个 PN 结 IV，从 0V 扫到 5V，初始步长 5V，最小步长 1.25V，最多 3 次 attempt，最多 3 轮" \
  --loop-no-llm
```

This writes:

```text
runs/task_plans/planner_smoke/
  task_plan_result.json
  task.json
```

`--loop-no-llm` means the later autonomous execution loop should not use LLM diagnosis. It does not disable the planner itself.

## Execute With LLM Planning

Ask the LLM planner to create `TaskSpec`, then run the autonomous loop:

```bash
python3.11 -m tcad_agent.tools.task_runner \
  --planner llm \
  --task-id planner_execute_smoke \
  --text "做一个 PN 结 IV，从 0V 扫到 5V，初始步长 5V，最小步长 1.25V，最多 3 次 attempt，最多 3 轮" \
  --execute \
  --no-llm
```

Here:

- `--planner llm`: use the model to understand the user task;
- `--no-llm`: do not use the model during execution-time diagnosis;
- `--use-llm`: use the model during execution-time diagnosis too.

## No Fallback

By default, planner failures fall back to deterministic parsing. To require a valid LLM-generated plan:

```bash
python3.11 -m tcad_agent.tools.task_planner \
  --task-id planner_strict \
  --text "PN junction IV from 0 to 5 V step 5 V" \
  --no-fallback
```

For `task_runner`, use:

```bash
python3.11 -m tcad_agent.tools.task_runner \
  --planner llm \
  --no-planner-fallback \
  --task-id planner_strict_run \
  --text "PN junction IV from 0 to 5 V step 5 V" \
  --execute \
  --no-llm
```

## Safety Rails

The planner repair layer currently:

- extracts JSON from wrapped text;
- accepts `task_spec`, `task`, or raw schema objects;
- maps aliases such as `sweep.step` to `sweep.step_v`;
- maps parameter aliases such as `geometry.length_um`, `doping.p`, and `mesh.contact_mesh`;
- maps `execution.attempts` to `execution.max_attempts`;
- normalizes supported values to `simulate_iv`, `pn_junction`, and `devsim`;
- adjusts `min_step_v` if it exceeds `step_v`;
- adjusts `parameters.junction_um` if it falls outside the device length;
- ignores unsupported fields;
- falls back to deterministic parsing unless strict mode is requested.

## Parameterized Planning

The planner can also emit device parameters:

```bash
python3.11 -m tcad_agent.tools.task_planner \
  --task-id param_planner \
  --text "做 PN 结 IV，0 到 0.2V，步长 0.1V，器件长度 0.2um，结位置 0.08um，P 区掺杂 1e17，N 区掺杂 2e17，温度 325K" \
  --loop-no-llm \
  --no-fallback
```

The current executable surface remains deliberately narrow: PN junction IV with DEVSIM. More device types should be added by extending `TaskSpec`, adding a concrete tool, and then allowing the planner to emit the new schema branch.

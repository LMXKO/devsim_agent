# TCAD Run Queue

`tcad_agent.tools.run_queue` is the durable experiment-management layer for long-running TCAD work.

It stores queued runs in SQLite, leases work to workers, and keeps enough state to resume after process crashes or machine restarts. Built-in executable items include `supervisor` and `mission_agent`, so a queue entry can represent either one delegated TCAD action or a long-horizon autonomous mission.

## What It Provides

- global task database in `runs/run_queue.sqlite`;
- run queue with priority ordering;
- worker leases for concurrency control and duplicate-run prevention;
- pause / resume / cancel controls;
- stale lease recovery for long jobs interrupted mid-run;
- per-item attempts, checkpoints, result JSON, result state path, and failure reason;
- resource budget fields (`budget_seconds`, `budget_cases`) for future scheduling policies;
- history listing and filtering by status or tool.
- polling daemon mode for long-running background execution with idle and stop-file exits.
- default runners for mission/supervisor, device tools, convergence, sweeps, optimization, benchmarks, engineering objectives, and reports.

## Enqueue A Supervisor Goal

```bash
python3.11 -m tcad_agent.tools.run_queue enqueue \
  --queue-id q_mosfet_goal \
  --tool supervisor \
  --goal "做 2D MOSFET Id-Vg gate_start 0V gate_stop 1V gate_step 0.25V drain_voltage 0.05V" \
  --priority 10 \
  --max-attempts 2 \
  --budget-seconds 3600
```

Equivalent JSON request form:

```bash
python3.11 -m tcad_agent.tools.run_queue enqueue \
  --tool supervisor \
  --request-json '{"goal_text":"做 PN 二极管 breakdown 从 0V 到 -5V step 0.5V","max_cycles":3,"execute":true}'
```

## Enqueue A Long-Horizon Mission

Use `mission_agent` when the queue item should refresh history, delegate to the supervisor, repair suspicious runs, and write a conclusion.

```bash
python3.11 -m tcad_agent.tools.run_queue enqueue \
  --queue-id q_autonomous_mosfet \
  --tool mission_agent \
  --request-json '{"goal_text":"做 MOSFET Id-Vg，检查 mesh/model convergence，修复失败点，最后给工程结论","use_llm_decomposer":true,"allow_llm_fallback":true,"execute":true}' \
  --priority 20 \
  --max-attempts 2 \
  --budget-seconds 14400
```

The `mission_agent` queue runner accepts `use_llm_decomposer`/`use_llm` and `allow_llm_fallback`/`no_llm_fallback`, then passes those through to the mission DAG executor.

## Browser Workbench

The preferred interactive surface is the local web app:

```bash
python3.11 -m tcad_agent.tools.web_app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` to enqueue missions, start/stop a worker, pause/resume/cancel queue items, and run an LLM health check from the page.

## Run A Worker

```bash
python3.11 -m tcad_agent.tools.run_queue worker \
  --owner tcad_worker_1 \
  --concurrency 1 \
  --lease-seconds 7200
```

The worker claims queued items by priority, marks them `running`, executes them, then records `completed` or `failed`.

Concurrency is controlled by leases. Multiple workers can point at the same queue database; each item is claimed by at most one active owner.

## Run A Daemon

For longer autonomous sessions, run a polling daemon instead of a one-shot worker:

```bash
python3.11 -m tcad_agent.tools.run_queue daemon \
  --owner tcad_daemon_1 \
  --concurrency 1 \
  --lease-seconds 7200 \
  --poll-interval-seconds 10 \
  --max-idle-loops 60 \
  --stop-file runs/queue.stop
```

The daemon repeatedly runs the worker, sleeps when no item is available, and exits when the stop file appears, the max loop count is reached, or the configured idle loop budget is exhausted.

## Inspect History

```bash
python3.11 -m tcad_agent.tools.run_queue list --limit 20
python3.11 -m tcad_agent.tools.run_queue list --status failed
python3.11 -m tcad_agent.tools.run_queue show q_mosfet_goal
```

For completed TCAD artifacts, rebuild and query the experiment index as the second memory layer:

```bash
python3.11 -m tcad_agent.tools.experiment_index --rebuild --root runs
python3.11 -m tcad_agent.tools.experiment_index --list --limit 20
```

## Pause, Resume, Cancel

```bash
python3.11 -m tcad_agent.tools.run_queue pause q_mosfet_goal
python3.11 -m tcad_agent.tools.run_queue resume q_mosfet_goal
python3.11 -m tcad_agent.tools.run_queue cancel q_mosfet_goal
```

Pause and cancel clear the worker lease. A running simulation process is not force-killed by this command yet; the current implementation gives the durable control state needed for cooperative workers.

## Recover Interrupted Work

```bash
python3.11 -m tcad_agent.tools.run_queue recover
```

Expired `running` leases are returned to `queued` if attempts remain. If `max_attempts` has been reached, the item becomes `failed`.

Workers call stale recovery before claiming new work, so normal operation does not require a separate recovery command.

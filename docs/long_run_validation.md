# Long-Run Validation

`tcad_agent.tools.long_run_validation` runs an unattended validation of the long-horizon execution stack.

It exercises:

- run queue enqueue;
- queue daemon polling;
- executable tool dispatch;
- physical benchmark generation;
- experiment index rebuild;
- durable `validation_state.json`.

Run:

```bash
python3.11 -m tcad_agent.tools.long_run_validation \
  --validation-id smoke_longrun
```

By default it queues compact Schottky and photodiode extended-device runs. Custom queue items can be supplied with `--queue-goals-json`.

This is not a substitute for multi-day production soak testing, but it is the fast regression harness that proves the long-running control path is wired end to end.

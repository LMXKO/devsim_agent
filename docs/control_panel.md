# TCAD Control Panel

`tcad_agent.tools.control_panel` generates a static HTML/JSON control panel for a TCAD runs root.

For the interactive browser page that can submit missions and control workers, use [web_app.md](web_app.md).

It summarizes:

- run queue items;
- experiment index records;
- physical benchmark statuses;
- long-run validation states.
- configured LLM endpoint/model status.

Run:

```bash
python3.11 -m tcad_agent.tools.control_panel \
  --root runs \
  --output-dir runs/control_panel
```

Outputs:

```text
runs/control_panel/index.html
runs/control_panel/control_panel.json
```

The panel is static and can be opened directly in a browser. It is meant as the first operations surface for long-running autonomous TCAD work.

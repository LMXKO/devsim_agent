# TCAD Setup

The project needs a real TCAD simulator installed locally. The first supported simulator should be DEVSIM, which is distributed as a Python package and can be controlled directly from Python scripts.

## Local Status

This repository does not include simulator binaries or proprietary TCAD software. Install DEVSIM and the Python dependencies in your own local environment before running simulation tasks.

Optional tools not installed yet:

- `gmsh`: useful later for more complex geometry and mesh generation;
- `klayout`: useful later for layout/GDS inspection and workflows;
- `langgraph`: useful later if the agent state machine needs a durable orchestration framework.

## Recommended Environment

Use Python 3.11 for the first prototype instead of the system default Python. A fixed Python version makes simulator behavior, packages, and future benchmark runs easier to reproduce.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install devsim numpy pandas matplotlib optuna pydantic pyyaml
```

If you prefer installing directly with `pip3.11`, the Tsinghua PyPI mirror also works:

```bash
pip3.11 install --user -i https://pypi.tuna.tsinghua.edu.cn/simple devsim numpy pandas matplotlib optuna pydantic pyyaml
```

## First Verification

After installation, verify that DEVSIM can be imported:

```bash
source .venv/bin/activate
python -c "import devsim; print('devsim import ok')"
```

Matplotlib may try to write font/cache files under the home directory. In restricted execution environments, set writable cache paths before plotting:

```bash
mkdir -p .cache/matplotlib .cache/fontconfig
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
```

## What Should Be Installed First

For the first milestone, only install the minimum stack:

- `devsim`: TCAD simulator;
- `numpy`: numerical processing;
- `pandas`: tables and sweep results;
- `matplotlib`: plots;
- `optuna`: parameter search;
- `pydantic`: structured task and run records;
- `pyyaml`: benchmark task files.

Additional open-source TCAD or EDA tools such as KLayout, ngspice, or OpenROAD should be added later only when the agent needs those capabilities.

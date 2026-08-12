<h1 align="center">TrunkX — run_3pg_main release</h1>

Minimal release of the 3PG forestry model, containing only what is needed
to run it via `scripts/run_3pg_main.py`.

## Setup

```bash
uv sync
```

The R comparison interface (`trunx.gp3.run_r3pg`) is not included in this
release. If you need it, install the required system packages
(`r-base-dev`, `libtirpc-dev`) and run:

```bash
uv sync --extra r-interface
```

## Running the model

Place your input workbook at `data/threepg_inputs/solling_data.xlsx`
(not included — supply your own; see `src/trunx/gp3/PG3_model_impl.py`'s
`prepare_data` for the expected `site`/`climate`/`species`/`parameters`/
`observed` sheet structure), then run:

```bash
uv run python scripts/run_3pg_main.py
```

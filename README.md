<h1 align="center">TrunkX</h1>

![logo](./images/logo.png)


## Development Setup

After cloning the repo, install the pre-commit hooks:

```bash
uvx pre-commit install
```

This runs `ruff check`, `ruff format`, and `ty check` automatically before each commit.

The project can then be installed by running

```bash
uv sync
```

Note that by default, the interface to the R implementation of 3gp is not included.
To install the required dependencies, first make sure that the packages
`r-base-dev` and `libtirpc-dev` are installed, then run:

```bash
uv sync --extra r-interface
```


## Data Collection

- **ICOS**: can be fetched using a python API, see [example](examples/ICOS_download.py).

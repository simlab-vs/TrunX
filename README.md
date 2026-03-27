<h1 align="center">TrunkX</h1>

![logo](./images/logo.png)


## Development Setup

After cloning the repo, install the pre-commit hooks:

```bash
uvx pre-commit install
```

This runs `ruff check`, `ruff format`, and `ty check` automatically before each commit.

## Data Collection

- **ICOS**: can be fetched using a python API, see [example](examples/ICOS_download.py).

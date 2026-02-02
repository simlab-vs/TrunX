# Project template

This template should be used for every Python project in the lab. It uses:

- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- [`ruff`](https://docs.astral.sh/ruff/) for code formatting.
- [`pyright`](https://github.com/microsoft/pyright) for type checking.
- [pre-commit](https://pre-commit.com/) hooks for automated validation.

## Dependency management

We use [`uv`](https://docs.astral.sh/uv/) for dependency management. It is just as
full-featured as `poetry`, but much faster. Follow the instructions below to
create a new project:

1. Update the name of the project in `pyproject.toml`.
2. Change the name of the folder `src/python-base` to match the project name.
3. Run `uv sync` from the root of the repo. 
This will create a virtual environment and install needed development dependencies.
4. Add the dependencies you need (and run this same command every time you need
   a new package):

   ```sh
   uv add polars lightgbm
   ```

5. Take a look at the `uv`'s [Getting started guide](https://docs.astral.sh/uv/getting-started/).

## Pre-commit hooks

First, install [pre-commit](https://pre-commit.com/):

```sh
uv add install pre-commit
```

Then, install the pre-commit hooks:

```sh
uv run pre-commit install
```

This will create a `.git/hooks/pre-commit` file that will run the pre-commit
hooks every time you commit. Upon the first commit, the hooks will be installed.

Some hooks output error message that require a manual change (e.g., linting
errors). Other hooks perform automated fixes. Either way, you need to re-run
the commit command:

```sh
git commit -m "My message"
```

### Code formatting

Among the pre-commit hooks, you will find one that runs
[`ruff`](https://docs.astral.sh/ruff/) on every Python file. It is also warmly
recommended that you set up `ruff` in your IDE (e.g., Visual Studio Code, PyCharm).

### Typing

We recommend the use of [type hints](https://docs.python.org/3/library/typing.html)
of your code. One of the pre-commit hooks is [`pyright`](https://github.com/microsoft/pyright),
which will perform type checking when hints are available. This reduces greatly the
risk of bugs and the maintainability of the code.


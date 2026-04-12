# Contributing to pylcp

## Setup

Install dependencies with [Poetry](https://python-poetry.org/):

```bash
poetry install --with dev
```

## Code quality

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [pyright](https://github.com/microsoft/pyright) for type checking.

### Linting

```bash
poetry run ruff check pylcp
```

### Formatting

```bash
poetry run ruff format pylcp
```

All code should be formatted with `ruff format` before committing. The formatter uses a line length of 100 (configured in `pyproject.toml`).

### Type checking

```bash
poetry run pyright
```

Pyright is configured in basic mode and checks the `pylcp/` directory. The `gratings` module is excluded from type checking due to an optional `numba` dependency.

## Running tests

```bash
poetry run pytest
```

GPU tests are skipped automatically when no CUDA device is available.

## Style notes

- **Line length**: 100 characters.
- **Imports**: sorted by `ruff` (isort rules). `jax.config.update()` calls before `jax.numpy` imports are expected and exempt from import-order checks.
- **Variable names**: single-letter names like `l`, `I`, `O` are standard physics notation and allowed.
- **Docstrings**: NumPy style. Use `r"""` for docstrings containing LaTeX math (e.g. `\rho`, `\mu`).
- **Lambdas**: short lambda expressions are acceptable for simple physics formulas.
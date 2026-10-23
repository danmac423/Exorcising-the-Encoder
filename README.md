# my_project

Python project template (uv + ruff + ty + pytest + pre-commit + GitHub Actions).

## Start a new project

1. On GitHub: **Use this template** (or `git clone` and delete `.git`, then `git init`).
2. Rename `my_project` to your project name (lowercase, underscores) in:
   - the `src/my_project/` directory,
   - `pyproject.toml` (`name`, `[project.scripts]`, `description`),
   - `Makefile` (`run` target),
   - `tests/test_main.py` (imports),
   - this README.
3. `uv lock && make install`

## Commands

| Command          | What it does                      |
| ---------------- | --------------------------------- |
| `make install`   | `uv sync` and install git hooks   |
| `make fmt`       | format and auto-fix lint          |
| `make check`     | lint + type check + tests         |
| `make run`       | run the package                   |
| `make clean`     | remove caches and build artifacts |

## Layout

```
src/my_project/
  main.py         entry point (`main()`), wired in pyproject [project.scripts]
  config.py       constants
  services/       sub-package: greeter.py, formatting.py
tests/            pytest tests
```

## Imports

- Across the package: absolute, e.g. `from my_project.config import APP_NAME`.
- Between siblings in one package: relative, e.g. `from .formatting import shout`.
- Tests import the installed package the same way (`from my_project.services.greeter import greet`).
- Every folder with code needs an `__init__.py` (can be empty).

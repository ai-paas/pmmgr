# Repository Guidelines

## Project Structure & Module Organization

`pmmgr/` contains the Python service. `main.py` defines the FastAPI routes and command-line entry points, `db.py` owns the SQLite model metadata layer, `cm.py` manages Ollama and llama.cpp chat models, and `common.py` provides settings, enums, and test helpers. Runtime configuration lives in `configs/`. CSV fixtures and SQLite databases are under `data/`; treat `data/db/pmmgr.db` as application data, not a disposable test fixture. Shell helpers are in `scripts/`, while `docs/` holds project references and benchmark artifacts.

## Build, Test, and Development Commands

Use Python 3.10 or newer, preferably in a virtual environment.

```bash
python -m pip install -r requirements.txt
./scripts/run.sh
python pmmgr/main.py init_pminfo --csv_file data/pminfo.csv
PYTHONPATH=.:./pmmgr python -m unittest pmmgr.db
PYTHONPATH=.:./pmmgr python -m unittest pmmgr.cm
```

`run.sh` starts Uvicorn on port 8001. The initialization command reloads model metadata from CSV and mutates the configured database. The database tests use `data/db/test.db`; model-pool tests may require local Ollama/llama.cpp models and should be treated as integration tests. `scripts/test_get.sh` and `scripts/test_post.sh` are manual curl examples for a separately configured service.

## Coding Style & Naming Conventions

Follow PEP 8: four-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for constants. Keep route handlers thin and put persistence or model-loading behavior in `db.py` or `cm.py`. Add type hints to public functions and Pydantic fields. No formatter or linter is configured, so keep imports grouped, remove unused imports, and match nearby code.

## Testing Guidelines

Tests use the standard-library `unittest` framework and currently live beside implementation code in classes ending with `Test`. Name methods `test_<behavior>`. Isolate database tests with `settings.TEST_DB_PATH`, clean generated files in `tearDown`, and mock external model calls where practical. There is no enforced coverage target; cover new CRUD branches, validation failures, and API error handling.

## Commit & Pull Request Guidelines

History uses brief subjects such as `v0.2` and Korean feature summaries. Prefer a concise imperative subject that names the change, for example `모델 메타데이터 검증 추가`. Keep commits focused. Pull requests should explain behavior and data/schema impact, list verification commands, link relevant issues, and include sample requests/responses for API changes. Never commit credentials, private keys, `.env` files, logs, generated answers, or test databases.

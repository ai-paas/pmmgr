# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PMMgr (Pretrained Model Manager) is a FastAPI service for the AIPaaS platform that manages pre-trained HuggingFace models — primarily lightweight Ollama models — and exposes them through a REST API. It tracks model metadata in SQLite and provides a load-balanced chat inference layer via LangChain.

## Commands

```bash
# Install dependencies (Python 3.10+)
python -m pip install -r requirements.txt

# Start the server (Uvicorn on port 8001)
./scripts/run.sh

# Initialize model metadata database from CSV
python pmmgr/main.py init_pminfo --csv_file data/pminfo.csv

# Run database CRUD tests (unit-level, uses data/db/test.db)
PYTHONPATH=.:./pmmgr python -m unittest pmmgr.db

# Run chat-model pool tests (integration-level, requires local Ollama models)
PYTHONPATH=.:./pmmgr python -m unittest pmmgr.cm

# Run a single test method
PYTHONPATH=.:./pmmgr python -m unittest pmmgr.db.DBTest.test_insert_and_get_pretrained_model
```

## Architecture

```
pmmgr/
├── main.py      # FastAPI app, routes, CLI entry points (run_server / init_pminfo)
├── cm.py        # ChatModelPool singleton, MultiChatModel, ModelWrapper
├── db.py        # PretrainedModelInfo Pydantic model + SQLite CRUD + tests
├── common.py    # Settings (pydantic-settings), enums, FileUtils, BaseTest
configs/
├── logging.conf # Python logging config for the 'pmmgr' logger
├── msg.json     # User-facing error message strings keyed by MessageCode
data/
├── pminfo.csv   # Production model metadata seed (loaded by init_pminfo)
├── db/pmmgr.db  # Production SQLite database — treat as application data
scripts/
├── run.sh       # Sets PYTHONPATH and starts the server
```

**Key architectural patterns:**

- **ChatModelPool** (`cm.py:124`) is a singleton. On init, it reads all `PretrainedModelInfo` rows from the DB and creates one `MultiChatModel` per model ID, each containing N `ModelWrapper` replicas (where N = `n_srv_instances`). This provides basic load-balancing: when no wrapper is ready, it retries with a 2-second delay up to 10 times.
- **ModelWrapper** (`cm.py:29`) wraps a LangChain `BaseChatModel` and tracks a `ready` boolean that gates concurrent access. Only Ollama (`ChatOllama`) and llama.cpp (`LlamaCpp`) backends are implemented; others raise `NotImplementedError`.
- **DB layer** (`db.py`) uses raw `sqlite3.connect` context managers — no ORM. The `PretrainedModelInfo` Pydantic model handles serialization with `from_row_dict()` / `to_dict()`, including type coercion for dates, JSON, and numerics.
- **Settings** use `pydantic-settings` (`common.py:14`), loaded from `.env` if present. Hardcoded defaults point to absolute paths under `/trunk/pmmgr/`.

## API Routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Health check |
| GET | `/model/chat/{id_or_path}/{query}` | Run chat inference on a model |
| POST | `/model/info/` | Register a new model |
| GET | `/model/info/{id_or_path}` | Get one model's metadata |
| GET | `/model/info/` | List all models |
| PUT | `/model/info/` | Update model metadata |
| DELETE | `/model/info/{id_or_path}` | Remove a model |

## Testing

Tests use `unittest` and live in the same file as implementation code (classes suffixed with `Test`, e.g. `DBTest`, `ChatModelPoolTest`, `MultiChatModelTest`). `BaseTest` in `common.py` suppresses warnings and configures logging. Database tests must use `settings.TEST_DB_PATH` and clean up in `tearDown`. The chat-model tests require local Ollama models listed in `data/test_pminfo.csv` — treat these as integration tests.

## Coding Style

PEP 8: 4-space indent, `snake_case` functions, `PascalCase` classes, `UPPER_CASE` constants. Route handlers should be thin; put persistence logic in `db.py` and model-loading in `cm.py`. Add type hints to public signatures and Pydantic fields. Group standard-library imports, then third-party, then local. No formatter or linter is configured.

## Key Constraints

- The `id_ed25519` file in the repo root is a private key — never commit it, read it into prompts, or expose it.
- Never commit `.env` files, logs (`*.log`), generated answer files, or test databases.
- `data/db/pmmgr.db` is application data, not a disposable fixture — do not delete it during development.
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

# Generate benchmark answers from all models (calls ollama CLI directly, not the API)
./scripts/gen_answer.sh
```

## Architecture

```
pmmgr/
├── main.py      # FastAPI app, routes, CLI entry points (run_server / init_pminfo)
├── cm.py        # ChatModelPool singleton, MultiChatModel (extends LangChain BaseChatModel), ModelWrapper
├── db.py        # PretrainedModelInfo Pydantic model + SQLite CRUD + DBTest
├── common.py    # Settings (pydantic-settings), enums, FileUtils, MessageCode, BaseTest
configs/
├── logging.conf # Python logging config for the 'pmmgr' logger
├── msg.json     # User-facing error message strings keyed by MessageCode enum values
data/
├── pminfo.csv   # Production model metadata seed (10 models, loaded by init_pminfo)
├── test_pminfo.csv  # Test fixture (3 models) — used by ChatModelPoolTest
├── db/pmmgr.db  # Production SQLite database — treat as application data, not a fixture
scripts/
├── run.sh       # Sets PYTHONPATH and starts the server
├── gen_answer.sh # Benchmarks all models via `ollama run` (bypasses the API)
```

**Key architectural patterns:**

- **ChatModelPool** (`cm.py:124`) is a singleton (via `get_instance()`). On init, it reads all `PretrainedModelInfo` rows from the DB and creates one `MultiChatModel` per model ID, each containing N `ModelWrapper` replicas (where N = `n_srv_instances`). It supports dynamic scaling at runtime via `increase_cm_srv_instances(id_or_path)` and `decrease_cm_srv_instances(id_or_path)`, which mutate the in-memory pool and persist the updated `n_srv_instances` count to the DB.

- **MultiChatModel** (`cm.py:28`) extends LangChain's `BaseChatModel` — this means it is itself a valid LangChain runnable, not just a container. It owns a list of `ModelWrapper` instances and load-balances across them via `get_ready_chat_model()`, which scans for a wrapper whose `ready` flag is `True`, retrying every 2 seconds up to 10 times.

- **ModelWrapper** (`cm.py:29`) is a thin inner class that wraps a LangChain `BaseChatModel` (e.g. `ChatOllama`) and tracks a `ready` boolean. `_generate` and `_stream` set `ready=False` before delegating and restore it in `finally`, gating concurrent access. Only Ollama (`ChatOllama`) and llama.cpp (`LlamaCpp`) backends are implemented; others raise `NotImplementedError`.

- **`config` field** on `PretrainedModelInfo` is a `dict` (JSON-serialized in SQLite). When a chat model is instantiated, `config` is unpacked as `**kwargs` to the LangChain model constructor (`ChatOllama` or `LlamaCpp`). This is the primary mechanism for passing model-specific parameters (temperature, top_p, etc.) without code changes.

- **DB layer** (`db.py`) uses raw `sqlite3.connect` context managers — no ORM. The `PretrainedModelInfo` Pydantic model handles serialization with `from_row_dict()` / `to_dict()`, including type coercion for dates, JSON, and numerics. CRUD functions all accept an optional `db_path` parameter (defaults to `settings.DB_PATH`).

- **Settings** use `pydantic-settings` (`common.py:14`), loaded from `.env` if present. Hardcoded defaults point to absolute paths under `/trunk/pmmgr/` — if the repo is cloned elsewhere, override via environment variables or edit `common.py`.

- **Error handling** uses `MessageCodeException` (`common.py:110`), which wraps a `MessageCode` enum and resolves user-facing messages from `configs/msg.json`. `MessageCode` values are also used directly as constants in `main.py` (e.g. `DB_ERROR`, `NO_AVAILABLE_CHAT_MODEL`).

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

Tests use `unittest` and live in the same file as implementation code (classes suffixed with `Test`, e.g. `DBTest` in `db.py`, `ChatModelPoolTest` and `MultiChatModelTest` in `cm.py`). `BaseTest` in `common.py` suppresses warnings and configures logging.

- **DB tests** (`db.py`): Use `settings.TEST_DB_PATH` (`data/db/test.db`). `setUp` creates a fresh table; `tearDown` deletes the DB file. These are fast, self-contained unit tests.
- **Chat-model tests** (`cm.py`): Require local Ollama models listed in `data/test_pminfo.csv` (3 models: deepseek-r1:1.5b, gemma3:1b, llama3.2:1b). `MultiChatModelTest.test_invoke` actually calls `ollama` — treat these as integration tests.
- **Patch pattern**: Tests that modify `ChatModelPool` state use `_use_test_db()` / `_restore_db()`, which monkey-patches `pmmgr.cm.db` to redirect all DB operations to the test database and resets the singleton. This is needed because `ChatModelPool` is a singleton that reads from the production DB by default.

## Coding Style

PEP 8: 4-space indent, `snake_case` functions, `PascalCase` classes, `UPPER_CASE` constants. Route handlers should be thin; put persistence logic in `db.py` and model-loading in `cm.py`. Add type hints to public signatures and Pydantic fields. Group standard-library imports, then third-party, then local. No formatter or linter is configured.

## Key Constraints

- Never commit `.env` files, logs (`*.log`), generated answer files, test databases, or private keys.
- `data/db/pmmgr.db` is application data, not a disposable fixture — do not delete it during development.
- The `Settings` defaults use absolute paths under `/trunk/pmmgr/`; override via `.env` if the repo is cloned to a different location.

## Coding Rules
### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]


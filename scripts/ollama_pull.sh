#!/bin/bash

ROOT_DIR=$(cd $(dirname $0)/..; pwd)
DB_PATH="$ROOT_DIR/data/db/pmmgr.db"

if [ ! -f "$DB_PATH" ]; then
    echo "DB not found: $DB_PATH"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "ollama command not found."
    exit 1
fi

# Get ollama-type model IDs from the DB
DB_MODELS=$(sqlite3 "$DB_PATH" "SELECT id_or_path FROM PretrainedModelInfo WHERE mtype='ollama';")

if [ -z "$DB_MODELS" ]; then
    echo "No ollama models found in DB."
    exit 0
fi

# Get already-pulled model names (first column, skip header)
PULLED_MODELS=$(ollama list | tail -n +2 | awk '{print $1}')

# Pull missing models
MISSING_COUNT=0
for model in $DB_MODELS; do
    if echo "$PULLED_MODELS" | grep -qxF "$model"; then
        echo "SKIP: $model (already pulled)"
    else
        echo "PULL: $model"
        ollama pull "$model"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ $MISSING_COUNT -eq 0 ]; then
    echo "All models already pulled."
else
    echo "Pulled $MISSING_COUNT model(s)."
fi
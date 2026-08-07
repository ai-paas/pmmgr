#!/bin/bash

ROOT_DIR=$(cd $(dirname $0)/..; pwd)
PORT=8001
BASE_URL="http://localhost:$PORT"
QUERY="Hello"
TIMEOUT=120
RESPONSE_FILE="/tmp/pmmgr_api_test_response.txt"

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
}
trap cleanup EXIT

# Start server
echo "Starting FastAPI server on port $PORT..."
cd "$ROOT_DIR"
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/pmmgr"
python pmmgr/main.py run_server --port $PORT &
SERVER_PID=$!

# Wait for readiness
echo -n "Waiting for server"
for i in $(seq 1 30); do
    if curl -s "$BASE_URL/" > /dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    sleep 1
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server failed to start."
    exit 1
fi

# Get ollama models from DB
MODELS=$(sqlite3 "$ROOT_DIR/data/db/pmmgr.db" "SELECT id_or_path FROM PretrainedModelInfo WHERE mtype='ollama';")

if [ -z "$MODELS" ]; then
    echo "No ollama models found in DB."
    exit 0
fi

# Test each model
TOTAL=0
PASS=0
FAIL=0

for model in $MODELS; do
    TOTAL=$((TOTAL + 1))

    # URL-encode ':' -> '%3A', '/' -> '%2F'
    ENCODED_MODEL=$(echo "$model" | sed 's/:/%3A/g; s/\//%2F/g')
    URL="$BASE_URL/model/chat/$ENCODED_MODEL/$QUERY"

    echo ""
    echo "[$TOTAL] Testing: $model"

    HTTP_CODE=$(curl -s -o "$RESPONSE_FILE" -w "%{http_code}" --max-time $TIMEOUT "$URL")
    RESPONSE=$(head -c 200 "$RESPONSE_FILE")

    if [ "$HTTP_CODE" = "200" ]; then
        echo "  PASS (200): $RESPONSE"
        PASS=$((PASS + 1))
    else
        echo "  FAIL ($HTTP_CODE): $RESPONSE"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "Results: $PASS/$TOTAL passed, $FAIL failed"
echo "=========================================="
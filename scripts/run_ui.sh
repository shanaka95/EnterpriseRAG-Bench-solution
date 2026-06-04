#!/usr/bin/env bash
# Launch the Streamlit UI for the reactive RAG agent.
#
# Usage:
#   ./scripts/run_ui.sh           # uses defaults (port 8599)
#   PORT=9000 ./scripts/run_ui.sh # custom port
#
# The script:
#   1. Activates the backend venv (which has streamlit, langgraph, etc.)
#   2. Loads MINIMAX_* env vars from .env if present
#   3. Starts streamlit on 0.0.0.0:<PORT>
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8599}"
HOST="${HOST:-0.0.0.0}"

# Load .env if present
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# Defaults for the agent
export MINIMAX_BASE_URL="${MINIMAX_BASE_URL:-https://api.minimax.io/anthropic}"
export MINIMAX_MODEL="${MINIMAX_MODEL:-MiniMax-M2.7}"
export MINIMAX_API_KEY="${MINIMAX_API_KEY:-}"

if [[ -z "${MINIMAX_API_KEY}" ]]; then
    echo "ERROR: MINIMAX_API_KEY is not set. Put it in .env or export it." >&2
    exit 1
fi

echo "Starting Streamlit on http://${HOST}:${PORT}"
echo "  model:     ${MINIMAX_MODEL}"
echo "  base_url:  ${MINIMAX_BASE_URL}"
exec ./backend/venv/bin/streamlit run ui/streamlit_app.py \
    --server.port "${PORT}" \
    --server.address "${HOST}" \
    --server.headless true \
    --browser.gatherUsageStats false

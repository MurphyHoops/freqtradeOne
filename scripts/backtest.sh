#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <timerange> [timeframe] [stake]" >&2
    exit 1
}

TIMERANGE=${1:-}
TIMEFRAME=${2:-5m}
STAKE=${3:-100}

if [[ -z "$TIMERANGE" ]]; then
    usage
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
else
    echo "docker compose command not found. Install Docker and retry." >&2
    exit 2
fi

echo "Running backtest: ${COMPOSE_CMD[*]} run --rm freqtrade backtesting -c user_data/configs/v29_backtest.json -s TaxBrainV30 -i $TIMEFRAME --timerange $TIMERANGE --stake-amount $STAKE"
"${COMPOSE_CMD[@]}" run --rm freqtrade backtesting \
  -c user_data/configs/v29_backtest.json \
  -s TaxBrainV30 -i "$TIMEFRAME" \
  --timerange "$TIMERANGE" \
  --stake-amount "$STAKE"

RESULTS_DIR="user_data/backtest_results"
TARGET_DIR="user_data/runs/last"
mkdir -p "$TARGET_DIR"

shopt -s nullglob
files=("$RESULTS_DIR"/*.json)
if (( ${#files[@]} == 0 )); then
    echo "Warning: no backtest result files found in $RESULTS_DIR" >&2
else
    for f in "${files[@]}"; do
        cp "$f" "$TARGET_DIR"/
    done
    echo "Copied ${#files[@]} result file(s) to $TARGET_DIR"
fi

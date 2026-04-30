#!/bin/bash
set -euo pipefail

# --- Config ---
ENV_FILE="scripts/spot_detection_pipeline/environment.yml"

# --- Helpers ---
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- Initiate conda ---
module load EBModules
module load Anaconda3


# --- Parse ENV name ---
ENV_NAME=$(grep -m1 '^name:' "$ENV_FILE" | awk '{print $2}')
[[ -z "$ENV_NAME" ]] && die "Could not parse 'name:' from $ENV_FILE"
log "Target enviroment: $ENV_NAME"

# --- Check if ENV exist & create/update if not ---
env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

if env_exists; then
    log "Enviroment '$ENV_NAME' found. Checking packages..."

    if conda compare --name "$ENV_NAME" "$ENV_FILE" &>dev/null; then
        log "Enviroment is up to date. Skipping install."
    else
        log "Package mismatch detected. Updating enviroment..."
        conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
        log "Enviroment updated."
    fi
else
    log "Enviroment '$ENV_NAME' not found. Creating..."
    conda env create --file "$ENV_FILE"
    log "Enviroment created."

fi

# --- Activate ENV ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME$

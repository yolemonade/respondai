#!/usr/bin/env bash
# Download symbolic music datasets used in RespondAI.
#
# Usage:
#   bash scripts/download_data.sh nottingham        # ~10MB, instant
#   bash scripts/download_data.sh lakh              # ~1.6GB, slow
#   bash scripts/download_data.sh all
#
# Files land in ./datasets/<name>/. Idempotent: if the target directory
# already exists and is non-empty, the script skips that dataset.

set -euo pipefail

DEST=${RESPONDAI_DATA_DIR:-./datasets}
mkdir -p "$DEST"

download_nottingham() {
    local dir="$DEST/nottingham"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "[nottingham] already present at $dir, skipping."
        return
    fi
    mkdir -p "$dir"
    echo "[nottingham] downloading..."
    # Mirror used by Boulanger-Lewandowski et al.; redirects to a Zenodo
    # mirror nowadays. If this URL breaks, replace with any of the standard
    # Nottingham mirrors.
    curl -L -o "$dir/nottingham_midi.zip" \
        "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip"
    cd "$dir"
    unzip -q nottingham_midi.zip
    rm nottingham_midi.zip
    cd - > /dev/null
    echo "[nottingham] done."
}

download_lakh() {
    local dir="$DEST/lakh"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "[lakh] already present at $dir, skipping."
        return
    fi
    mkdir -p "$dir"
    echo "[lakh] downloading (this is ~1.6GB)..."
    # Lakh MIDI Dataset, "clean" subset (smaller, easier to filter).
    curl -L -o "$dir/lmd_clean.tar.gz" \
        "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
    echo "[lakh] extracting..."
    cd "$dir"
    tar -xzf lmd_clean.tar.gz
    rm lmd_clean.tar.gz
    cd - > /dev/null
    echo "[lakh] done."
}

case "${1:-all}" in
    nottingham) download_nottingham ;;
    lakh)       download_lakh ;;
    all)
        download_nottingham
        download_lakh
        ;;
    *)
        echo "Unknown dataset: $1" >&2
        echo "Usage: $0 {nottingham|lakh|all}" >&2
        exit 1
        ;;
esac

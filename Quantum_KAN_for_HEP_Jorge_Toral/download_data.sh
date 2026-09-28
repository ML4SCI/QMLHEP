#!/bin/bash

# ==============================================================================
# DATA DOWNLOAD PIPELINE
# ==============================================================================
# Usage: ./download_data.sh [--with-qg]
#   (default)   top tagging dataset only (Zenodo record 2603256) -> data/raw/top
#   --with-qg   also download the quark-gluon dataset (Zenodo record 19362155)
#               -> data/raw/quark-gluon
# Safe to re-run: aria2 resumes partial downloads and skips completed ones.

WITH_QG=0
for arg in "$@"; do
    case "$arg" in
        --with-qg) WITH_QG=1 ;;
        -h|--help)
            sed -n '6,10p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (usage: $0 [--with-qg])" >&2
            exit 2
            ;;
    esac
done

# Locate the root directory of the local repository (where download_data.sh is located)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define immutable raw data paths according to the workspace design
RAW_TOP="$REPO_ROOT/data/raw/top"
RAW_QG="$REPO_ROOT/data/raw/quark-gluon"

mkdir -p "$RAW_TOP"
if [ $WITH_QG -eq 1 ]; then mkdir -p "$RAW_QG"; fi

echo "System dependencies validation..."
echo "--------------------------------------------------"

# Runtime Dependency enforcement
if ! command -v aria2c &> /dev/null; then
    echo "Missing required tool: aria2. Installing..."
    sudo apt update && sudo apt install -y aria2
    if [ $? -ne 0 ]; then
        echo "Automated setup failed. Please install aria2 manually."
        exit 1
    fi
fi

echo "--------------------------------------------------"
echo "Populating raw data matrices using accelerated download threads..."
echo "--------------------------------------------------"

# Create a secure temporary workspace for orchestration mapping
F_TEMPORAL=$(mktemp)

# ==============================================================================
# Top Tagging Datasets
# ==============================================================================
# 'out=' gives each file its clean name directly (no '?download=1' suffix)
for split in train val test; do
    echo "https://zenodo.org/records/2603256/files/${split}.h5?download=1" >> "$F_TEMPORAL"
    echo "  dir=$RAW_TOP" >> "$F_TEMPORAL"
    echo "  out=${split}.h5" >> "$F_TEMPORAL"
done

# ==============================================================================
# Quark-Gluon Datasets (optional, --with-qg)
# ==============================================================================
if [ $WITH_QG -eq 1 ]; then
    for i in 0 1 2; do
        echo "https://zenodo.org/records/19362155/files/QG_jets_fp32_${i}.npz?download=1" >> "$F_TEMPORAL"
        echo "  dir=$RAW_QG" >> "$F_TEMPORAL"
        echo "  out=QG_jets_fp32_${i}.npz" >> "$F_TEMPORAL"
    done
fi

# Run aria2c reading from the unified mapped file configuration
# -c  : Resume any partially completed downloads, skipping fully completed ones gracefully
# --auto-file-renaming=false : Prevent downloading duplicate files with suffixes like .1, .2
# -j2 : Restrict parallel file execution to prevent connection dropping
# -x4 : Safe threshold limit per host for CERN/Zenodo protection
aria2c -c --auto-file-renaming=false -j2 -x4 -s4 --no-netrc -i "$F_TEMPORAL"
ESTADO=$?
rm "$F_TEMPORAL"

echo "--------------------------------------------------"
if [ $ESTADO -eq 0 ]; then
    echo "Portability setup completed. Data environment is synchronized!"
else
    echo "Download finished with errors (aria2c exit code $ESTADO); re-run to resume."
fi
echo "--------------------------------------------------"
exit $ESTADO

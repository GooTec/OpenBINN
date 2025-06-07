#!/bin/bash
# Download prostate dataset from Zenodo and extract under ../data/prostate
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
ZIP_PATH="${DATA_DIR}/prostate.zip"
URL="https://zenodo.org/records/15614376/files/prostate.zip?download=1&preview=1"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/prostate" ]; then
  echo "Downloading prostate dataset..."
  wget -O "$ZIP_PATH" "$URL"
  echo "Extracting..."
  unzip -o "$ZIP_PATH" -d "$DATA_DIR"
  rm "$ZIP_PATH"
else
  echo "Dataset already exists at $DATA_DIR/prostate"
fi

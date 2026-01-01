#!/usr/bin/env bash
#
# Setup script for ProjectRanni training environment
# Usage: ./setup_pod.sh
#

set -euo pipefail

# Configuration
REPO_URL="https://github.com/dhmnr/ProjectRanni.git"
DATASET_URL="https://s3.us-east-1.amazonaws.com/project-ranni-raw-data-v0.1/margit/margit_100_256x144.zarr.zip"
PROJECT_DIR="$HOME/ProjectRanni"
DATASET_DIR="$PROJECT_DIR/dataset"

echo "==> Setting up ProjectRanni training environment..."

# Clone repository if not already present
if [[ -d "$PROJECT_DIR" ]]; then
    echo "==> Project directory already exists, pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull
else
    echo "==> Cloning repository..."
    cd "$HOME"
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if ! command -v uv &> /dev/null; then
        source "$HOME/.local/bin/env"
    fi
else
    echo "==> uv already installed"
fi

# Install dependencies
echo "==> Syncing dependencies..."
uv sync
uv pip install 'jax[cuda12_local]'

# Download dataset if not already present
if [[ -d "$DATASET_DIR/margit_100_256x144.zarr" ]]; then
    echo "==> Dataset already exists, skipping download"
else
    echo "==> Downloading dataset..."
    mkdir -p "$DATASET_DIR"
    cd "$DATASET_DIR"

    if [[ -f "margit_100_256x144.zarr.zip" ]]; then
        echo "==> Zip file exists, extracting..."
    else
        wget "$DATASET_URL"
    fi

    unzip -o margit_100_256x144.zarr.zip
    cd "$PROJECT_DIR"
fi

# Configure Weights & Biases
echo "==> Configuring Weights & Biases..."
uv run wandb login

echo "==> Setup complete!"

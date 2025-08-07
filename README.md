# README

## 🔧 Environment Setup Instructions

```bash
# Install UV (Rust-based Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with Python 3.11
uv venv --python=3.11 ~/hmbenv

# Activate
source ~/hmbenv/bin/activate

# Install Python dependencies
uv pip install -r requirements.txt

```

## Structure

* data_utils: contains dataset classes
* models: model definitions
* notebooks: miscellaneous notebooks I've created when first understanding datasets, disregard
* sbatch: sbatch files and bash files for cluster running
* scripts: entry points for command line, sbatch, and bash files
* trainers: library of functions for training, finetuning, and testing

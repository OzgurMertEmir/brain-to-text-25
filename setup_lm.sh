#!/usr/bin/env bash
set -euo pipefail

echo "=== Brain-to-Text LM environment setup ==="
echo "=== Takes a while to run (~15-20 mins) ==="

# 1. Ensure we are at the project root (where setup_lm.sh lives)
if [[ ! -f "setup_lm.sh" ]]; then
  echo "This script must be run from the root directory of the project (where setup_lm.sh is)."
  exit 1
fi

# 2. Install Miniconda (if not already present)
MINICONDA_DIR="$HOME/miniconda3"
if [[ -d "$MINICONDA_DIR" ]]; then
  echo "[info] Miniconda already installed at $MINICONDA_DIR"
else
  echo "[info] Installing Miniconda under $MINICONDA_DIR ..."
  wget -O Miniconda3-latest-Linux-x86_64.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

  # -b: batch (no prompts), -p: install location
  bash ./Miniconda3-latest-Linux-x86_64.sh -b -p "$MINICONDA_DIR"
  rm -f Miniconda3-latest-Linux-x86_64.sh
  echo "[info] Miniconda installed."
fi

# 3. Initialize conda for this shell
echo "[info] Initializing conda ..."
source ~/miniconda3/etc/profile.d/conda.sh

# 4. Accept Anaconda Terms of Service for the required channels
#    This avoids CondaToSNonInteractiveError during setup_lm.sh
echo "[info] Accepting Anaconda Terms of Service for main and r channels ..."
# Optional shortcut: auto-accept; still run explicit commands as well
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

# These may fail if the plugin is not present yet, so don't abort on failure
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 5. Install system build dependencies
echo "[info] Installing system build dependencies (cmake, build-essential) ..."
sudo apt-get update -y
sudo apt-get install -y cmake build-essential

# 6. Clean old LM build artifacts so setup_lm.sh doesn't abort
echo "[info] Removing previous LM build directories (if any) ..."
rm -rf language_model/runtime/server/x86/build
rm -rf language_model/runtime/server/x86/fc_base

# 7. Run the original LM setup script
echo "[info] Running setup_lm.sh to create b2txt25_lm and build lm_decoder ..."

# Ensure that the script is run from the root directory of the project
if [ ! -f "setup_lm.sh" ]; then
    echo "This script must be run from the root directory of the project."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/build directory does not exist
if [ -d "language_model/runtime/server/x86/build" ]; then
    echo "The language_model/runtime/server/x86/build directory already exists. Please remove it before running this script."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/fc_base directory does not exist
if [ -d "language_model/runtime/server/x86/fc_base" ]; then
    echo "The language_model/runtime/server/x86/fc_base directory already exists. Please remove it before running this script."
    exit 1
fi

# make sure CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "CMake is not installed. Please install CMake >= 3.14 before running this script with 'sudo apt-get install cmake'."
    exit 1
fi

# make sure gcc is installed
if ! command -v gcc &> /dev/null; then
    echo "GCC is not installed. Please install GCC >= 10.1 before running this script with 'sudo apt-get install build-essential'."
    exit 1
fi

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment with Python 3.9
conda create -n b2txt25_lm python=3.9 -y

# Activate the new environment
conda activate b2txt25_lm

# Upgrade pip
pip install --upgrade pip

# Install additional packages
pip install \
    torch==1.13.1 \
    redis==5.0.6 \
    jupyter==1.1.1 \
    numpy==1.24.4 \
    matplotlib==3.9.0 \
    scipy==1.11.1 \
    scikit-learn==1.6.1 \
    tqdm==4.66.4 \
    g2p_en==2.1.0 \
    omegaconf==2.3.0 \
    huggingface-hub==0.23.4 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    accelerate==0.33.0 \
    bitsandbytes==0.41.1 \
    pandas==2.3.3 \
    editdistance==0.8.1 \
    h5py==3.14.0 

# cd to the language model directory and install the language model
cd language_model/runtime/server/x86
python setup.py install

# cd back to the root directory
cd ../../../..

echo
echo "Setup complete! Verify it worked by activating the conda environment with the command 'conda activate b2txt25_lm'."
echo


echo "=== LM environment setup complete ==="
echo "You can now do: 'conda activate b2txt25_lm' and run the LM tools."

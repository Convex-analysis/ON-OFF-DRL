#!/bin/bash

echo "Choose environment type:"
echo "1) Conda"
echo "2) Python venv"
read -p "Enter 1 or 2: " env_choice

if [ "$env_choice" == "1" ]; then
    # Conda environment
    ENV_NAME="mamba_scheduler_env"
    echo "Creating Conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.10
    conda activate $ENV_NAME
    conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
    pip install mamba-ssm numpy matplotlib tqdm
    echo "Conda environment '$ENV_NAME' is ready. Activate it with: conda activate $ENV_NAME"
elif [ "$env_choice" == "2" ]; then
    # Python venv
    ENV_DIR="venv_mamba_scheduler"
    echo "Creating Python venv in: $ENV_DIR"
    python -m venv $ENV_DIR
    source $ENV_DIR/bin/activate || source $ENV_DIR/Scripts/activate
    pip install --upgrade pip
    pip install torch mamba-ssm numpy matplotlib tqdm
    echo "Python venv '$ENV_DIR' is ready. Activate it with: source $ENV_DIR/bin/activate (or $ENV_DIR\\Scripts\\activate on Windows)"
else
    echo "Invalid choice."
    exit 1
fi
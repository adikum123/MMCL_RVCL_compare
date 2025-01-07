#!/bin/bash
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o outs/100k.out
#SBATCH -e errs/100k.err

# Variables
CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"
WORKING_DIR="MMCL_RVCL_compare"
PYTHON_VERSION="3.7.9"
VENV_NAME="mmcl_rvcl"
TORCH_URL="https://download.pytorch.org/whl/torch-1.6.0%2Bcu101-cp37-cp37m-linux_x86_64.whl"

echo "Creating and starting the container..."
enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
enroot start --mount $(pwd):/workspace -- $CONTAINER_NAME

cd $WORKING_DIR

if ! command -v pyenv &>/dev/null; then
    echo "Installing pyenv and pyenv-virtualenv..."
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
fi

if ! pyenv versions | grep -q $PYTHON_VERSION; then
    echo "Installing Python $PYTHON_VERSION with pyenv..."
    pyenv install $PYTHON_VERSION
fi

if ! pyenv virtualenvs | grep -q $VENV_NAME; then
    echo "Creating virtual environment '$VENV_NAME' with pyenv-virtualenv..."
    pyenv virtualenv $PYTHON_VERSION $VENV_NAME
fi

echo "Activating virtual environment '$VENV_NAME'..."
pyenv activate $VENV_NAME

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch 1.6.0 manually..."
pip install $TORCH_URL

echo "Installing dependencies from requirements.txt..."
pip install -r $REQUIREMENTS_FILE

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Running the Python script..."
python $PYTHON_SCRIPT --kernel_type rbf

echo "Deactivating virtual environment..."
pyenv deactivate

echo "Script completed."

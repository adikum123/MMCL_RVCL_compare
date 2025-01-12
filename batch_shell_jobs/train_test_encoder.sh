#!/bin/bash
set +e  # Disable exit on error

# Variables
CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"
WORKING_DIR="workspace/MMCL_RVCL_compare"
PYTHON_VERSION="3.7"
TORCH_URL="https://download.pytorch.org/whl/torch-1.6.0%2Bcu101-cp37-cp37m-linux_x86_64.whl"

echo "Creating and starting the container..."
# Check if the container exists
if ! enroot list | grep -q "^$CONTAINER_NAME$"; then
    echo "Container $CONTAINER_NAME does not exist. Creating it..."
    enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
else
    echo "Container $CONTAINER_NAME already exists. Skipping creation."
fi
enroot start --root --mount $(pwd):/workspace $CONTAINER_NAME <<EOF
    [ -d ~/.pyenv ] && rm -rf ~/.pyenv
    echo "Installing pyenv and pyenv-virtualenv..."
    apt update
    apt install git -y
    apt install curl -y
    apt install -y build-essential gcc make libffi-dev zlib1g-dev libssl-dev libreadline-dev libbz2-dev libsqlite3-dev
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"

    if ! pyenv versions | grep -q $PYTHON_VERSION; then
        echo "Installing Python $PYTHON_VERSION with pyenv..."
        pyenv install $PYTHON_VERSION
    fi
    pyenv global $PYTHON_VERSION

    cd $WORKING_DIR

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing dependencies from requirements.txt..."
    pip install -r $REQUIREMENTS_FILE

    export PYTHONPATH=$(pwd):$PYTHONPATH

    echo "Running the Python script..."
    python $PYTHON_SCRIPT --kernel_type rbf --encoder_num_iters 1000 --linear_eval_num_iters 200 --encoder_lr 1e-4 --svm_lr 1e-4 --linear_eval_lr 1e-4

    echo "Script completed."
EOF
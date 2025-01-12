#!/bin/bash
set +e  # Disable exit on error

# Variables
CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"
WORKING_DIR="/workspace/MMCL_RVCL_compare"  # Ensure this is absolute

echo "Creating and starting the container..."

# Check if the container exists
if ! enroot list | grep -q "^$CONTAINER_NAME$"; then
    echo "Container $CONTAINER_NAME does not exist. Creating it..."
    enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
else
    echo "Container $CONTAINER_NAME already exists. Skipping creation."
fi

# Start the container and run commands
echo "Starting container"
enroot start --root --mount $(pwd):/workspace $CONTAINER_NAME <<'EOF'
    set +e  # Stop execution inside the container if any command fails

    PYTHON_VERSION=3.7.17

    # Install prerequisites for pyenv
    apt update
    apt install -y git curl build-essential gcc make libffi-dev zlib1g-dev \
                libssl-dev libreadline-dev libbz2-dev libsqlite3-dev libncurses5-dev \
                libgdbm-dev libnss3-dev liblzma-dev tk-dev uuid-dev

    # Configure pyenv environment
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"

    # Install pyenv
    rm -rf /root/.pyenv
    echo "Installing pyenv..."
    curl https://pyenv.run | bash

    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"

    # Install and activate Python version
    if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
        echo "Installing Python $PYTHON_VERSION..."
        pyenv install 3.7.17
    else
        echo "Python $PYTHON_VERSION already exists"
    fi

    pyenv global $PYTHON_VERSION

    # start a virtual enviroment with pyenv version
    pyenv virtualenv $PYTHON_VERSION mmcl_rvcl_venv
    pyenv activate mmcl_rvcl_venv

    # Move to the working directory
    cd $WORKING_DIR

    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    # Set Python path and run the script
    export PYTHONPATH=$(pwd):$PYTHONPATH
    echo "Running the Python script..."
    python $PYTHON_SCRIPT --kernel_type rbf --encoder_num_iters 1000 --linear_eval_num_iters 200 --encoder_lr 1e-4 --svm_lr 1e-4 --linear_eval_lr 1e-4

    echo "Script completed."
EOF

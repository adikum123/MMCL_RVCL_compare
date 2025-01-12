#!/bin/bash
set +e  # Disable exit on error

# Variables
CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"
WORKING_DIR="/workspace/MMCL_RVCL_compare"  # Ensure this is absolute
PYTHON_VERSION="3.7"

echo "Creating and starting the container..."

# Check if the container exists
if ! enroot list | grep -q "^$CONTAINER_NAME$"; then
    echo "Container $CONTAINER_NAME does not exist. Creating it..."
    enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
else
    echo "Container $CONTAINER_NAME already exists. Skipping creation."
fi

# Start the container and run commands
enroot start --root --mount $(pwd):/workspace $CONTAINER_NAME <<'EOF'
    set -e  # Stop execution inside the container if any command fails

    # Install prerequisites
    apt update
    apt install -y python3.7 python3.7-venv python3.7-distutils curl

    # Verify Python 3.7 installation
    echo "Verifying Python 3.7 installation..."
    python3.7 --version || { echo "Python 3.7 not found. Exiting."; exit 1; }

    # Create and activate a virtual environment
    echo "Creating and activating virtual environment..."
    python3.7 -m venv /workspace/venv
    source /workspace/venv/bin/activate

    # Move to the working directory
    cd $WORKING_DIR

    # Ensure pip is installed and upgraded
    echo "Installing and upgrading pip..."
    curl https://bootstrap.pypa.io/get-pip.py | python

    echo "Installing dependencies from requirements.txt..."
    pip install -r $REQUIREMENTS_FILE

    # Save dependencies (optional, for sharing environments)
    pip freeze > requirements.txt

    # Set Python path and run the script
    export PYTHONPATH=$(pwd):$PYTHONPATH
    echo "Running the Python script..."
    python $PYTHON_SCRIPT --kernel_type rbf --encoder_num_iters 1000 --linear_eval_num_iters 200 --encoder_lr 1e-4 --svm_lr 1e-4 --linear_eval_lr 1e-4

    echo "Script completed."
EOF

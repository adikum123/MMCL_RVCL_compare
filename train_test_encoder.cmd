#!/bin/bash
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o outs/100k.out
#SBATCH -e errs/100k.err

CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"
WORKING_DIR="MMCL_RVCL_compare"
VENV_DIR="venv"

echo "Creating and starting the container..."
enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
enroot start --mount $(pwd):/workspace -- $CONTAINER_NAME

cd $WORKING_DIR

echo "Setting up Python 3.7 virtual environment..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y \
    wget build-essential software-properties-common \
    python3.7 python3.7-dev python3.7-venv

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3.7 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r $REQUIREMENTS_FILE

echo "Installing CUDA toolkit..."
apt-get install -y cuda-toolkit-11-8
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Running the Python script..."
python $PYTHON_SCRIPT --kernel_type rbf

deactivate

echo "Script completed."

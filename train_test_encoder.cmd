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

echo "Creating and starting the container..."
enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
enroot start --mount $(pwd):/workspace -- $CONTAINER_NAME

cd ..
cd MMCL_RVCL_compare/

echo "Updating package lists..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y \
    wget build-essential software-properties-common

echo "Adding Python 3.7 repository..."
add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -y

echo "Installing Python 3.7..."
apt-get install -y python3.7 python3.7-dev python3.7-venv
ln -sf /usr/bin/python3.7 /usr/bin/python3
ln -sf /usr/bin/python3.7 /usr/bin/python

echo "Installing pip..."
wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
python3 get-pip.py && rm get-pip.py

echo "Installing CUDA toolkit..."
apt-get install -y cuda-toolkit-11-8
echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && ldconfig

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r $REQUIREMENTS_FILE

export PYTHONPATH=/workspace/$WORKING_DIR:$PYTHONPATH

echo "Running the Python script..."
python $PYTHON_SCRIPT --kernel_type rbf

echo "Script completed."

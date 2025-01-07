#!/bin/bash
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o outs/100k.out
#SBATCH -e errs/100k.err

enroot remove $CONTAINER_NAME

CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"

enroot import docker://ubuntu
echo "After enroot import"
echo ($ls)

echo "Creating and starting conatiner"
enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
enroot start --m MMCL_RVCL_compare/ $CONTAINER_NAME
cd ..
cd MMCL_RVCL_compare/

echo "Installing necessary dependencies"
apt-get update && apt-get install -y \
    wget build-essential software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update

echo "Installing python 3.7"
apt-get install -y python3.7 python3.7-dev python3.7-venv && \
    ln -s /usr/bin/python3.7 /usr/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/bin/python

echo "Installing pip"
wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

echo "Installing cuda"
apt-get install -y cuda-toolkit-11-8 && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && ldconfig

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Installing requirements"
pip install --upgrade pip
pip install -r $REQUIREMENTS_FILE

export PYTHONPATH=$(pwd):$PYTHONPATH

python $PYTHON_SCRIPT --kernel_type rbf

enroot remove $CONTAINER_NAME
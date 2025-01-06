#!/bin/bash
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o outs/100k.out
#SBATCH -e errs/100k.err

CONTAINER_IMAGE="nvidia+tensorflow+20.12-tf1-py3.sqsh"
CONTAINER_NAME="mmcl_rvcl"
PYTHON_SCRIPT="mmcl/train_test_encoder.py"
REQUIREMENTS_FILE="requirements.txt"

enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
enroot start --m MMCL_RVCL_compare/ $CONTAINER_NAME << EOF
    cd ..
    cd MMCL_RVCL_compare/
    pip install --upgrade pip
    pip install -r $REQUIREMENTS_FILE
    export PYTHONPATH=$(pwd):$PYTHONPATH
EOF

python $PYTHON_SCRIPT --kernel_type rbf

enroot remove $CONTAINER_NAME
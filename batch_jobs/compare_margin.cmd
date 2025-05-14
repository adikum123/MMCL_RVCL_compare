#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o outs/margin.out
#SBATCH -e outs/margin.out

#!/bin/bash
echo "Creating and starting the container..."

enroot list
echo "Removing container"
enroot remove --force mmcl_rvcl
echo "Container removed"
enroot list
echo "Creating new container"
enroot create --name mmcl_rvcl ../nvidia+tensorflow+20.12-tf1-py3.sqsh
echo "Container created"
echo "Starting container"
enroot start --mount $(pwd):/workspace mmcl_rvcl <<'EOF'
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip uninstall -y onnx
    yes | pip install onnx==1.9.0
    yes | pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt

    echo "Computing plots for svm margin"
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python -u svm_margin/compare_svm_margin_resnet.py \
        --dataset cifar-10 \
        --C 1 \
        --kernel_type rbf \
        --positives_per_class 20 \
        --num_negatives 1000 \
        --num_retries 10

EOF

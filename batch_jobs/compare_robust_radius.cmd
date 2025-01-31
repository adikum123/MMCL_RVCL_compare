#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o outs/robust_radius_100k.out
#SBATCH -e outs/robust_radius_100k.out

echo "Creating and starting the container..."

enroot list
echo "Removing existing container"
enroot remove --force mmcl_rvcl
echo "Container removed"
enroot list
echo "Creating new container"
enroot create --name mmcl_rvcl ../nvidia+tensorflow+20.12-tf1-py3.sqsh
echo "Container created"
echo "Starting container"

enroot start --mount $(pwd):/workspace mmcl_rvcl <<'EOF'
    echo "Setting up the environment..."

    # Upgrade pip
    pip install --upgrade pip

    # Uninstall potentially conflicting versions of ONNX
    pip uninstall -y onnx
    yes | pip install onnx==1.9.0

    # Install PyTorch 1.13.1 and compatible torchvision/torchaudio
    echo "Installing PyTorch 1.13.1 and compatible libraries..."
    yes | pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117

    # Install other required dependencies
    pip install -r requirements.txt

    echo "Verifying installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
    python -c "import torchaudio; print(f'Torchaudio version: {torchaudio.__version__}')"
    python -c "import onnx; print(f'ONNX version: {onnx.__version__}')"

    echo "Computing plots for robust radius..."
    python compare_robust_radius.py \
        --mmcl_model cifar_model_base \
        --mmcl_checkpoint models/mmcl/cifar_model_base_rbf_C_100.pkl \
        --rvcl_model cifar_model_base \
        --rvcl_checkpoint models/unsupervised/cifar10_base_adv4.pkl \
        --dataset cifar-10 \
        --class_sample_limit 50 \

EOF

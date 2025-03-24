#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=28:00:00
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

    echo "Computing plots for robust radius..."
    python -u compare_randomized_smoothing.py \
        --mmcl_model cnn_4layer_b \
        --mmcl_checkpoint models/mmcl/rbf/cnn_4layer_b_C_1_rbf_auto.pkl \
        --rvcl_model cnn_4layer_b_adv \
        --rvcl_checkpoint models/linear_evaluate/cifar10_cnn_4layer_b_adv2.pkl \
        --regular_cl_model cnn_4layer_b \
        --regular_cl_checkpoint models/regular_cl/regular_cl_cnn_4layer_b_bs_32_lr_1e-3.pkl \
        --dataset cifar-10 \
        --positives_per_class 10

EOF

#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH -o outs/robust_radius.out
#SBATCH -e outs/robust_radius.out

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
    python -u compare_robust_radius.py \
        --mmcl_model mmcl rbf \
        --mmcl_checkpoint models/mmcl/rbf/finetune_mmcl_cnn_4layer_b_C_1.0_bs_512_lr_0.0001.pkl \
        --rvcl_model adversarial cl \
        --rvcl_checkpoint models/regular_cl/finetune_adv_regular_cl_info_nce_bs_512_lr_0.001.pkl \
        --regular_cl_model cl info nce \
        --regular_cl_checkpoint models/regular_cl/finetune_regular_cl_info_nce_bs_512_lr_0.001.pkl \
        --dataset cifar-10 \
        --max_steps 100 \
        --positives_per_class 30 \
        --negatives_per_class 2 \
        --num_retries 3

EOF

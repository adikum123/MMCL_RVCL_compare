#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH -o outs/robust_radius_100k.out
#SBATCH -e outs/robust_radius_100k.out

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

    echo "Computing plots for robust radisu"
    python compare_robust_radius.py \
        --mmcl_model cifar_model_base \
        --mmcl_checkpoint models/mmcl/cifar_model_base_rbf_C_100.pkl \
        --rvcl_model cifar_model_base \
        --rvcl_checkpoint models/unsupervised/cifar10_base_adv4.pkl \
        --dataset cifar-10 \
        --sample_limit 30 \

EOF

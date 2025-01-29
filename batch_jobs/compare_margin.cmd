#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o outs/margin_100k.out
#SBATCH -e outs/margin_100k.out

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
    python compare_svm_margin.py \
        --mmcl_model cifar_model \
        --mmcl_checkpoint models/mmcl/mnist_model_base_rbf_C_100.pkl \
        --rvcl_model cifar_model \
        --rvcl_checkpoint models/unsupervised/mnist_base.pkl \
        --dataset mnist \
        --C 100 \
        --kernel_type rbf \
        --class_sample_limit 200 \

EOF

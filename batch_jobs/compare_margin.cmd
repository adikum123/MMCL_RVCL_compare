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
        --mmcl_model cnn_4layer_b \
        --mmcl_checkpoint models/mmcl/rbf/cnn_4layer_b_C_1_rbf_auto.pkl \
        --rvcl_model cnn_4layer_b \
        --rvcl_checkpoint models/unsupervised/cifar10_cnn_4layer_b.pkl \
        --dataset cifar-10 \
        --C 1 \
        --kernel_type rbf \
        --class_sample_limit 250

EOF

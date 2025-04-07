#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH -o outs/resnet_unsupervised.out
#SBATCH -e outs/resnet_unsupervised.out

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

    export PYTHONPATH=$(pwd):$PYTHONPATH
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Training unsupervised model"
    python -u train_models/train_resnet_unsupervised.py \
        --resnet_unsupervised_ckpt models/resnet_pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar \
        --dataset cifar-10 \
        --batch_size 512 \
        --num_iters 200 \
        --lr 1e-3 \
        --use_validation \
        --scheduler_gamma 0.5 \
        --finetune \
        --step_size 25

EOF

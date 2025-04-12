#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH -o outs/linear_eval_mmcl.out
#SBATCH -e outs/linear_eval_mmcl.out

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
    echo "Testing performance on linear eval"
    python -u train_models/train_linear_eval_mmcl.py \
        --batch_size 256 \
        --dataset cifar-10 \
        --use_validation \
        --num_iters 100 \
        --step_size 25 \
        --lr_encoder 1e-3 \
        --lr_classifier 1e-3 \
        --model cnn_4layer_b \
        --finetune \
        --scheduler_gamma 0.5 \
        --mmcl_checkpoint models/mmcl/rbf/cnn_4layer_b_C_1_rbf_auto_bs_32_lr_0.0001.pkl

EOF

#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o outs/mnist_100k.out
#SBATCH -e outs/mnist_100k.out

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
    echo "Training encoder"
    python train_mmcl.py \
        --model_save_name mnist_cnn_4layer_b_linear \
        --model mnist_cnn_4layer_b \
        --dataset mnist \
        --batch_size 32 \
        --kernel_type linear \
        --num_iters 200 \
        --use_validation \
        --lr 1e-5 \
        --step_size 50 \
        --C 100

    if [ $? -eq 0 ]; then
        echo "Testing performance on linear eval"
        python train_linear_eval.py \
            --batch_size 32 \
            --dataset mnist \
            --use_validation \
            --num_iters 100 \
            --step_size 30 \
            --lr 1e-3 \
            --model mnist_cnn_4layer_b \
            --mmcl_checkpoint models/mmcl/poly/mnist_cnn_4layer_b_linear.pkl \
            --adv_img
    else
        echo "Training encoder failed, skipping linear evaluation."
    fi

EOF
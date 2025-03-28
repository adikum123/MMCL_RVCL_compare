#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH -o outs/cifar_C_1.out
#SBATCH -e outs/cifar_C_1.out

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
        --model_save_name cnn_4layer_b_C_1_poly_deg_3 \
        --model cnn_4layer_b \
        --dataset cifar-10 \
        --batch_size 32 \
        --kernel_type poly \
        --num_iters 200 \
        --lr 1e-4 \
        --use_validation \
        --step_size 50 \
        --C 1

    if [ $? -eq 0 ]; then
        echo "Testing performance on linear eval"
        python -u train_linear_eval_mmcl.py \
            --batch_size 32 \
            --dataset cifar-10 \
            --use_validation \
            --num_iters 100 \
            --step_size 30 \
            --lr 1e-3 \
            --model cnn_4layer_b \
            --relu_layer \
            --clean \
            --mmcl_checkpoint models/mmcl/rbf/cnn_4layer_b_C_1_poly_deg_3.pkl
    else
        echo "Training encoder failed, skipping linear evaluation."
    fi

EOF

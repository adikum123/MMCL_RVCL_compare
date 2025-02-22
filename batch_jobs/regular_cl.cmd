#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o outs/regular_cl_100k.out
#SBATCH -e outs/regular_cl_100k.out

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
    python train_regular_cl.py \
        --model_save_name regular_cl_cnn_4layer_b_bs_32_lr_1e-3 \
        --model cnn_4layer_b \
        --dataset cifar-10 \
        --use_validation \
        --batch_size 32 \
        --num_iters 200 \
        --lr 1e-3 \
        --step_size 50

    if [ $? -eq 0 ]; then
        echo "Testing performance on linear eval"
        python -u train_linear_eval_regular_cl.py \
            --batch_size 32 \
            --dataset cifar-10 \
            --use_validation \
            --num_iters 100 \
            --step_size 30 \
            --lr 1e-3 \
            --model cnn_4layer_b \
            --regular_cl_checkpoint models/regular_cl/regular_cl_cnn_4layer_b_bs_32_lr_1e-3.pkl \
            --adv_img
    else
        echo "Training encoder failed, skipping linear evaluation."
    fi

EOF
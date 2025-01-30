#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -o outs/cifar_100k.out
#SBATCH -e outs/cifar_100k.out

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
    python train_encoder.py \
        --model_save_name cifar_model_wide_poly_C_100_deegre_3 \
        --model cifar_model_wide \
        --dataset cifar-10 \
        --batch_size 32 \
        --kernel_type poly \
        --deegre 3 \
        --num_iters 200 \
        --lr 1e-6 \
        --use_validation \
        --step_size 50 \
        --C 100 \

echo "Testing performance on linear eval"
    python train_linear_eval.py \
        --batch_size 32 \
        --dataset cifar-10 \
        --use_validation \
        --num_iters 100 \
        --step_size 30 \
        --lr 1e-4 \
        --model cifar_model_deep \
        --load_checkpoint models/mmcl/cifar_model_wide_poly_C_100_deegre_3.pkl \
        --adv_img \

EOF

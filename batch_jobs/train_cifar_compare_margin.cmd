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
        --model_save_name cifar_model_base_rbf_C_100 \
        --model cifar_model \
        --dataset cifar-10 \
        --batch_size 32 \
        --kernel_type rbf \
        --num_iters 200 \
        --lr 1e-6 \
        --kernel_gamma 0.1 \
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
        --model cifar_model \
        --load_checkpoint models/mmcl/cifar_model_base_rbf_C_100.pkl \
        --adv_img \

    echo "Computing plots for svm margin"
    python compare_svm_margin.py \
        --mmcl_model cifar_model \
        --mmcl_checkpoint models/mmcl/cifar_model_base_rbf_C_100.pkl \
        --rvcl_model cifar_model \
        --rvcl_checkpoint models/unsupervised/cifar10_base_adv4.pkl \
        --dataset cifar-10 \
        --C 100 \
        --kernel_type rbf \
        --kernel_gamma 0.1 \
        --class_sample_limit 1000 \

EOF

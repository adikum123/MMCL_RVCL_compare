#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH -o outs/mnist_adv_100k.out
#SBATCH -e outs/mnist_adv_100k.out

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
    pip uninstall onnx
    pip install onnx==1.9.0 --force-reinstall --yes
    pip install -r requirements.txt

    export PYTHONPATH=$(pwd):$PYTHONPATH
    echo "Running the Python script..."
    python train_test_encoder.py \
        --model mnist_model_deep \
        --dataset mnist \
        --batch_size 64 \
        --kernel_type rbf \
        --encoder_num_iters 500 \
        --linear_eval_num_iters 200 \
        --encoder_lr 1e-3 \
        --step_size 30 \
        --scheduler_gamma 0.1 \
        --svm_lr 1e-3 \
        --linear_eval_lr 1e-3 \
        --C 20 \
        --adv_img

    echo "Script completed."

EOF
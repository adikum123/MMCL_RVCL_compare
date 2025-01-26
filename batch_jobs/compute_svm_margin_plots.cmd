#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH -o outs/cifar_adv_100k.out
#SBATCH -e outs/cifar_adv_100k.out

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
enroot start --mount $(pwd):/workspacesa_su  <<'EOF'
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip uninstall onnx
    pip install onnx==1.9.0
    pip install -r requirements.txt

    export PYTHONPATH=$(pwd):$PYTHONPATH
    echo "Running the Python script..."
    python compare_svm_margin.py \
        --mmcl_model cifar_model_deep \
        --mmcl_checkpoint models/mmcl/cifar_model_deep_C_20.0_kernel_type_rbf_train_data_clean.pkl \
        --rvcl_model cifar_base \
        --rvcl_checkpoint models/unsupervised/cifar10_base_adv4.pkl \
        --C 20 \
        --kernel_type rbf \
        --class_sample_limit 1000

    echo "Script completed."

EOF
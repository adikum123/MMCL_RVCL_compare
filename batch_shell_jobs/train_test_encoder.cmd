#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH -o outs/100k.out
#SBATCH -e errs/100k.err

#!/bin/bash
echo "Creating and starting the container..."

enroot remove mmcl_rvcl
enroot create --name mmcl_rvcl ../nvidia+tensorflow+20.12-tf1-py3.sqsh
echo "Starting container"
enroot start --root --mount $(pwd):/workspace mmcl_rvcl

python --version

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH=$(pwd):$PYTHONPATH
echo "Running the Python script..."
python mmcl/train_test_encoder.py --batch_size 256 --kernel_type rbf --encoder_num_iters 500 --linear_eval_num_iters 200 --encoder_lr 1e-4 --svm_lr 1e-4 --linear_eval_lr 1e-4

echo "Script completed."


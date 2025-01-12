#!/bin/bash
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH -o outs/100k.out
#SBATCH -e errs/100k.err

#!/bin/bash
set +e  # Disable exit on error

# Variables
CONTAINER_IMAGE="ubuntu.sqsh"
CONTAINER_NAME="mmcl_rvcl"

echo "Creating and starting the container..."

# Check if the container exists
if ! enroot list | grep -q "^$CONTAINER_NAME$"; then
    echo "Container $CONTAINER_NAME does not exist. Creating it..."
    enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE
else
    echo "Container $CONTAINER_NAME already exists. Skipping creation."
fi

# Start the container and run commands
echo "Starting container"
enroot start --root --mount $(pwd):/workspace $CONTAINER_NAME
set +e  # Stop execution inside the container if any command fails

# Install prerequisites for pyenv
apt update
apt install -y git curl build-essential gcc make libffi-dev zlib1g-dev \
            libssl-dev libreadline-dev libbz2-dev libsqlite3-dev libncurses5-dev \
            libgdbm-dev libnss3-dev liblzma-dev tk-dev uuid-dev
apt install python3.12-venv

# Configure pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# Install pyenv
rm -rf /root/.pyenv
echo "Installing pyenv..."
curl https://pyenv.run | bash

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"


echo "Installing Python 3.7.17..."
pyenv install 3.7.17
pyenv global 3.7.17

# start a virtual enviroment with pyenv version
echo "Creating and activating virtual enviroment"
pyenv virtualenv 3.7.17 mmcl_rvcl_venv
pyenv activate mmcl_rvcl_venv

# Move to the working directory
cd workspace/MMCL_RVCL_compare

python --version

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Set Python path and run the script
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "Running the Python script..."
python mmcl/train_test_encoder.py --batch_size 512 --kernel_type rbf --encoder_num_iters 1000 --linear_eval_num_iters 200 --encoder_lr 1e-4 --svm_lr 1e-4 --linear_eval_lr 1e-4

echo "Script completed."


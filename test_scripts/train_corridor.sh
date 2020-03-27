#!/bin/bash
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-0:00:00  # DD-HH:MM:SS

# Setup venv
cd $SLURM_TMPDIR
module load python/3.6 cuda cudnn
virtualenv --no-download env
source env/bin/activate
pip install --no-index -r ~/CuriousNetwork/nn/requirements.txt
pip install --no-index torch_gpu

# Copy code and data
cd $SLURM_TMPDIR
cp -r ~/CuriousNetwork/nn/ .
cd nn/
cp ~/CuriousNetwork/dataset.tar .
tar -xf dataset.tar

cd $SLURM_TMPDIR/CuriousNetwork/nn/

python train.py --train_path $SLURM_TMPDIR/CuriousNetwork/nn/dataset/jpg/corridor_train \
                --val_path $SLURM_TMPDIR/CuriousNetwork/nn/dataset/jpg/corridor_val \
                --test_path $SLURM_TMPDIR/CuriousNetwork/nn//dataset/jpg/corridor_test "$@"
#!/bin/bash
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-24:00:00  # DD-HH:MM:SS


SLURM_TMPDIR='./output'

if [ ! -d "$SLURM_TMPDIR" ]
then
  mkdir $SLURM_TMPDIR
fi

# Setup venv
cd $SLURM_TMPDIR
virtualenv --no-download env
source ~/.venv_ift725/bin/activate
pip install --no-index -r ~/repos/CuriousNetwork_fixes/requirements.txt

# Copy code and data
cd $SLURM_TMPDIR
cp -r ~/repos/CuriousNetwork/nn/ .
cd nn/
cp ~/repos/CuriousNetwork/dataset.tar .
tar -xf dataset.tar
cp ~/CuriousNetwork_fixes/barbar.py .

cd $SLURM_TMPDIR/nn/

# Start the training
python train.py --train_path $SLURM_TMPDIR/nn/dataset/jpg/corridor_train \
                --val_path $SLURM_TMPDIR/nn/dataset/jpg/corridor_val \
                --test_path $SLURM_TMPDIR/nn/dataset/jpg/corridor_test "$@"

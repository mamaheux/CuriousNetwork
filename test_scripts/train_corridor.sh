#!/bin/bash
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-24:00:00  # DD-HH:MM:SS

CUR_DIR=$(pwd)
SLURM_TMPDIR='./output'

if [ ! -d "$SLURM_TMPDIR" ]
then
  echo 'creating slur tmpdir'
  mkdir $SLURM_TMPDIR
fi

# Setup venv
cd $SLURM_TMPDIR
source ~/.venv_ift725/bin/activate
pip install --no-index -r ~/repos/CuriousNetwork_fixes/requirements.txt

# Copy code and data
if [ ! -d "./nn" ]
then
  echo 'copying nn to slurm tmpdir'
  cp -r ~/repos/CuriousNetwork/nn/ .
fi

cd nn/

if [ ! -e "./dataset.tar" ]
then
  echo 'copying dataset to slurm tmpdir'
  cp ~/repos/CuriousNetwork/dataset.tar .
fi

if [ ! -d "./dataset" ]
then
  echo 'extracting dataset'
  tar -xf dataset.tar
fi

cp ~/repos/CuriousNetwork_fixes/barbar.py .

# Start the training
echo 'initiating training'
python train.py --train_path /home/ggs22/repos/CuriousNetwork/test_scripts/output/nn/dataset/jpg/corridor_train \
                --val_path /home/ggs22/repos/CuriousNetwork/test_scripts/output/nn/dataset/jpg/corridor_val \
                --test_path /home/ggs22/repos/CuriousNetwork/test_scripts/output/nn/dataset/jpg/corridor_test "$@"

cd $CUR_DIR
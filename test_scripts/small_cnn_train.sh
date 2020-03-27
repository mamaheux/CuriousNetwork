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
cp -r ~/CuriousNetwork/nn ./
cd nn
cp ~/CuriousNetwork/data.tar .
tar xf data.tar

python train.py "$@"

OUTDIR=~/CuriousNetwork/out_300/lr_${LR}_unet_depth_${UNET_DEPTH}_unet_res_dense_size_${UNET_RES_DENSE_SIZE}_data_aug_${USE_DATA_AUG}_job_id_${SLURM_JOB_ID}
mkdir -p $OUTDIR
cp fig1.png $OUTDIR/
cp fig2.png $OUTDIR/
cp output.txt $OUTDIR/

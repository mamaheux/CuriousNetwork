#!/bin/bash
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-0:00:00  # DD-HH:MM:SS

LR=$1
UNET_DEPTH=$2
UNET_RES_DENSE_SIZE=$3
USE_DATA_AUG=$4

# Setup venv
cd $SLURM_TMPDIR
module load python/3.6 cuda cudnn
virtualenv --no-download env
source env/bin/activate
pip install --no-index -r ~/IFT725/IFT725-TP/TP3/requirements.txt
pip install --no-index torch_gpu

# Copy code and data
cd $SLURM_TMPDIR
cp -r ~/IFT725/IFT725-TP/TP3/ .
cd TP3
cp ~/data.tar.gz .
tar xf data.tar.gz

cd $SLURM_TMPDIR/TP3/src

if [ "$USE_DATA_AUG" = 1 ] ; then
    python train.py --model=IFT725UNet --num-epochs=300 --lr $LR --ift_725_unet_depth $UNET_DEPTH --ift_725_unet_res_dense_size $UNET_RES_DENSE_SIZE --data_aug --batch_size 3 > output.txt
else
    python train.py --model=IFT725UNet --num-epochs=300 --lr $LR --ift_725_unet_depth $UNET_DEPTH --ift_725_unet_res_dense_size $UNET_RES_DENSE_SIZE --batch_size 3 > output.txt
fi

OUTDIR=~/IFT725/out_300/lr_${LR}_unet_depth_${UNET_DEPTH}_unet_res_dense_size_${UNET_RES_DENSE_SIZE}_data_aug_${USE_DATA_AUG}_job_id_${SLURM_JOB_ID}
mkdir -p $OUTDIR
cp fig1.png $OUTDIR/
cp fig2.png $OUTDIR/
cp output.txt $OUTDIR/

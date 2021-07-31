#!/bin/bash
#YBATCH -r am_2
#SBATCH -N 1
#SBATCH -J sam_test
#SBATCH --output output/%j.out

# ======== Pyenv/ ========
# export PYENV_ROOT=$HOME/.pyenv
# export PATH=$PYENV_ROOT/bin:$PATH
# eval "$(pyenv init -)"

# ======== Modules ========
. /etc/profile.d/modules.sh
module load cuda/11.1 cudnn/cuda-11.1/8.0 nccl/cuda-11.1/2.7.8 openmpi/3.1.6

export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=3565

export NGPUS=2
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python save_model_WRN-28-10_with_SAM.py \
  --dist-url $MASTER_ADDR \
  --epochs 200 \
  --batch-size 8192 \
  --nbs 0.06 \
  --lr 0.2 \
  --experiment bs_8192_ngpus_2_lr_0.2_nbs_0.06

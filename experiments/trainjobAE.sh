#!/bin/bash

#SBATCH --job-name=trainAE
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 2-19:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 18              # Number of cores (-c)
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=120000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --ntasks=1
#SBATCH -o ./trainlogs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./trainlogs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=john.gkountouras@student.uva.nl

module load 2022
module load Anaconda3/2022.05

eval "$(conda shell.bash hook)"
conda activate biscuit

#python train_nf.py --data_dir /scratch-shared/gkounto/ithor --num_latents 40
python train_ae.py --data_dir /scratch-shared/gkounto/biscuit/data/ithor_extended \
                   --batch_size 64 \
                   --c_hid 64 \
                   --lr 2e-4 \
                   --warmup 1000 \
                   --num_latents 48 \
                   --regularizer_weight 4e-6 \
                   --max_epochs 100 \
                   --seed 42 \
                   --wandb

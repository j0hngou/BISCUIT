#!/bin/bash

#SBATCH --job-name=trainAE
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 1-12:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 8              # Number of cores (-c)
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
python train_frameonly_vae.py --data_dir /home/gkounto/BISCUIT/data_generation/data/gridworld_simplified_12c_3d/ --batch_size 64 --c_hid 64 --lr 2e-4 --warmup 1000 --num_latents 40 --cluster --max_epochs 300 --seed 42 --beta 1.0 --use_coordconv 
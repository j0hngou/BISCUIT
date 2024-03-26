#!/bin/bash

#SBATCH --job-name=trainMLP
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 0-02:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 16              # Number of cores (-c)
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

python train_mlp.py --data_dir /home/gkounto/BISCUIT/data_generation/data/gridworld_simplified_5c_3d/ --autoencoder_checkpoint /home/gkounto/BISCUIT/experiments/pretrained_models/AE_gridworld_simplified/AE_20l_64hid.ckpt --num_latents 20 --c_hid 128 --num_blocks 15 --lr 1e-3 --lr_text 1e-4 --num_samples 4 --batch_size 256 --warmup 100 --seed 42 --subsample_percentage 1.0 --max_epochs 400 --text --noise_level 0.03
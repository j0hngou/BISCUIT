#!/bin/bash

#SBATCH --job-name=trainAE
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 1-12:00         # Runtime in D-HH:MM, minimum of 10 minutes
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

#python train_nf.py --data_dir /scratch-shared/gkounto/ithor --num_latents 40
python train_nf.py --data_dir /scratch-shared/gkounto/biscuit/data/ithor/ \
                   --autoencoder_checkpoint /home/gkounto/BISCUIT/experiments/checkpoints/Autoencoder/AE_40l_64hid/version_0/checkpoints/AEITHOR.ckpt \
                   --num_latents 40 \
                   --c_hid 64 \
                   --num_flows 6 \
                   --lr 1e-3 \
                   --prior_action_add_prev_state \
                   --num_samples 2 \
                   --batch_size 1024 \
                   --warmup 100 \
                   --seed 42 \
		   --text True \
		   --lr_text 1e-4

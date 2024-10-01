#!/bin/bash

#SBATCH --job-name=trainNF
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 0-4:00         # Runtime in D-HH:MM, minimum of 10 minutes
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
conda activate biscuit-reasoners

# wandb agent orpheous1/llm-reasoners-pvt-examples_gridworld/cr8zafz2
python train_nf.py --data_dir /scratch-shared/gkounto/biscuit/data/ithor_new/ --autoencoder_checkpoint /home/gkounto/BISCUIT/experiments/pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt --num_latents 40 --c_hid 64 --num_flows 6 --lr 1e-3 --num_samples 2 --batch_size 64 --warmup 100 --seed 42 --text --lr_text 3e-3 --text_encoder siglip --subsample_percentage 1.0 --max_epochs 100 --seq_len 2 --num_workers 0 --text_only
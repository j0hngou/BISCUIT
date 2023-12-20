#!/bin/bash

#SBATCH --job-name=trainNF
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 0-12:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 12              # Number of cores (-c)
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
python train_nf.py --data_dir /scratch-shared/gkounto/biscuit/data/ithor_extended/ --autoencoder_checkpoint /scratch-shared/gkounto/biscuit/pretrained_models/ae_e/AE64_1/AE_64l_64hid.ckpt --c_hid 256 --num_flows 6 --num_samples 2 --flow_act_fn silu --hidden_per_var 32 --act_fn silu --num_latents 64 --prior_action_add_prev_state --logit_reg_factor 0.004 --lr 0.0005 --batch_size 1024 --num_workers 4 --seq_len 2 --max_epochs 200 --warmup 100 --text --lr_text 1e-3 --text_encoder siglip --subsample_percentage 1.0 --wandb --try_encodings True 
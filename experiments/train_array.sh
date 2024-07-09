#!/bin/bash

#SBATCH --job-name=QER
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 0-03:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 8              # Number of cores (-c)
#SBATCH --array=1-8
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=120000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --ntasks=1
#SBATCH -o ./trainlogs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./trainlogs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=john.gkountouras@student.uva.nl

module load 2022
module load Anaconda3/2022.05

eval "$(conda shell.bash hook)"
conda activate biscuit

source /home/gkounto/BISCUIT/data_generation/1.2.182.0/setup-env.sh

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" args.txt)

python train_nf.py $ARGS

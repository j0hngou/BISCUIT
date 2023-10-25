#!/bin/bash

chunk_size=36
num_sequences=400
# Make the prefix be the first argument to the script
prefix=$1
num_sequences=$2

for ((offset=0; offset<$num_sequences; offset+=$chunk_size)); do
    echo "Running chunk $offset"
    start=`date +%s`
    python parallel_data_generation_ithor.py --offset $offset --chunk_size $chunk_size --output_folder /scratch-shared/gkounto/biscuit/data/ithor_randomized_materials/ --prefix $prefix --randomize_materials --num_sequences $num_sequences
    echo "Done with chunk $offset out of $num_sequences"
    echo "-----------------------------------"
    end=`date +%s`
    runtime=$((end-start))
    echo "Elapsed time: $runtime"
    echo "-----------------------------------"
    echo "Remaining time: $((runtime * (num_sequences - offset) / 60)) minutes"
    echo "-----------------------------------"
done
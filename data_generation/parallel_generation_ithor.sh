#!/bin/bash

chunk_size=18
num_sequences=400
# Make the prefix be the first argument to the script
prefix=$1
num_sequences=$2

for ((offset=0; offset<$num_sequences; offset+=$chunk_size)); do
    echo "Running chunk $offset"
    start=`date +%s`
    python parallel_data_generation_ithor.py --offset $offset --chunk_size $chunk_size --output_folder /scratch-shared/gkounto/biscuit/data/ithor_extended/ --prefix $prefix --num_sequences $num_sequences
    echo "Done with chunk $offset out of $num_sequences"
    echo "-----------------------------------"
    end=`date +%s`
    runtime=$((end-start))
    echo "Elapsed time: $runtime"
    echo "-----------------------------------"
    remaining_sequences=$((num_sequences - offset))
    remaining_time=$((runtime * remaining_sequences / chunk_size / 60))
    echo "Remaining time: $remaining_time minutes"
    echo "-----------------------------------"
done
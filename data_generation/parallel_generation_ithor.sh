#!/bin/bash

chunk_size=18
num_sequences=100
# Make the prefix be the first argument to the script
prefix=$1
num_sequences=$2

existing_files=$(ls /scratch-shared/gkounto/biscuit/data/ithor_eval/${prefix}/*.npz | wc -l)
existing_samples=$((existing_files * 100))

# Calculate the starting offset and subtract 1 to avoid overlap
start_offset=$((existing_samples / 100))
start_offset=$((start_offset - 1))

# If start_offset is negative, set it to 0
if [ $start_offset -lt 0 ]; then
    start_offset=0
fi

for ((offset=start_offset; offset<$num_sequences; offset+=$chunk_size)); do
    echo "Running chunk $offset"
    start=`date +%s`
    python parallel_data_generation_ithor.py --offset $offset --chunk_size $chunk_size --output_folder /scratch-shared/gkounto/biscuit/data/ithor_eval/ --prefix $prefix --num_sequences $num_sequences --save_objects_states
    echo "Done with chunk $offset out of $num_sequences"
    echo "-----------------------------------"
    end=`date +%s`
    runtime=$((end-start))
    echo "Elapsed time: $runtime"
    echo "-----------------------------------"
    echo "Remaining time: $((runtime * (num_sequences - offset) / 60)) minutes"
    echo "-----------------------------------"
done
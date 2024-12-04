#!/bin/bash

# Function to run a chunk of experiments
run_chunk() {
    start=$1
    end=$2
    python models/Clustering/gmm_train.py $start $end
}

# Variables
total_start=2
total_end=10000
max_processes=32
max_chunk_size=100

# Calculate chunks
for ((i=total_start; i<=total_end; i+=max_chunk_size)); do
    chunk_start=$i
    chunk_end=$((i + max_chunk_size - 1))
    if [ $chunk_end -gt $total_end ]; then
        chunk_end=$total_end
    fi
    
    # Run in parallel, respecting max_processes
    ((count=count%max_processes)); ((count++==0)) && wait
    run_chunk $chunk_start $chunk_end &
done

# Wait for all background jobs to finish
wait

echo "All experiments completed."

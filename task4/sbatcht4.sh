#!/bin/bash
#
# Propagate environment variables to the compute node
#SBATCH --export=ALL
# Run in the standard partition (queue)
#SBATCH --partition=teaching
# Specify project account
#SBATCH --account=teaching
# Distribute processes in round-robin fashion, probs cap
#SBATCH --distribution=block:block
# No of cores required (max. of 16, 4GB RAM per core)
#SBATCH --ntasks=16
# Runtime (hard, HH:MM:SS)
#SBATCH --time=24:00:00
# Job name
#SBATCH --job-name=Definite_Integral_Compared
# Output file
#SBATCH --output=task4-slurm-%j.out
# Modify the line below to run your program:

# Run properly

perf stat -e cycles,instructions,cache-misses ./task4.py



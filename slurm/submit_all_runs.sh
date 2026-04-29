#!/bin/bash
# submit_all_runs.sh
# Submits Run 4a, 4b, 4c sequentially via SLURM job dependencies.
# Each run starts only if the previous one completed successfully.
#
# Usage: bash submit_all_runs.sh

set -e

SLURM_SCRIPT="run_ddpm.slurm"

# Helper: submit a job and extract the job ID from sbatch output
submit() {
    local output
    output=$(sbatch "$@")
    echo "$output" >&2
    echo "$output" | grep -oP '(?<=job )\d+'
}

# Run 4a — no attention
JOB1=$(submit --export=PARAM_FILE=configs/param_GZ2_run4a.json "$SLURM_SCRIPT")
echo "Submitted Run 4a (no attention):      job $JOB1"

# Run 4b — bottleneck only, starts after Run 4a
JOB2=$(submit --export=PARAM_FILE=configs/param_GZ2_run4b.json --dependency=afterok:$JOB1 "$SLURM_SCRIPT")
echo "Submitted Run 4b (bottleneck only):   job $JOB2"

# Run 4c — bottleneck + 16x16, starts after Run 4b
JOB3=$(submit --export=PARAM_FILE=configs/param_GZ2_run4c.json --dependency=afterok:$JOB2 "$SLURM_SCRIPT")
echo "Submitted Run 4c (attn 8+16):         job $JOB3"

echo ""
echo "All runs submitted. Monitor with: squeue -u $USER"
echo "Dependency chain: $JOB1 -> $JOB2 -> $JOB3"

#!/bin/bash
# submit_all_runs.sh


set -e

SLURM_SCRIPT="run_ddpm.slurm"
CONFIGS_DIR="../configs"

submit() {
    local output
    output=$(sbatch "$@")
    echo "$output" >&2
    echo "$output" | grep -oP '(?<=job )\d+'
}

JOB1=$(submit --export=PARAM_FILE=${CONFIGS_DIR}/param_GZ2_testA_noema_noasinh_attn8_16.json "$SLURM_SCRIPT")
echo "Submitted Test-A (no EMA, no asinh, attn=[8,16]):  job $JOB1"

JOB2=$(submit --export=PARAM_FILE=${CONFIGS_DIR}/param_GZ2_testB_ema_noasinh_attn8_16.json \
              --dependency=afterok:$JOB1 "$SLURM_SCRIPT")
echo "Submitted Test-B (+EMA, no asinh, attn=[8,16]):     job $JOB2"

JOB3=$(submit --export=PARAM_FILE=${CONFIGS_DIR}/param_GZ2_testC_ema_asinh_attn8_16.json \
              --dependency=afterok:$JOB2 "$SLURM_SCRIPT")
echo "Submitted Test-C (+EMA, +asinh, attn=[8,16]):       job $JOB3"

JOB4=$(submit --export=PARAM_FILE=${CONFIGS_DIR}/param_GZ2_testD_ema_asinh_noattn.json \
              --dependency=afterok:$JOB3 "$SLURM_SCRIPT")
echo "Submitted Test-D (+EMA, +asinh, attn=[]):           job $JOB4"

JOB5=$(submit --export=PARAM_FILE=${CONFIGS_DIR}/param_GZ2_testE_ema_asinh_attn8.json \
              --dependency=afterok:$JOB4 "$SLURM_SCRIPT")
echo "Submitted Test-E (+EMA, +asinh, attn=[8]):          job $JOB5"

echo ""
echo "All runs submitted. Monitor with: squeue -u $USER"
echo "Dependency chain: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5"

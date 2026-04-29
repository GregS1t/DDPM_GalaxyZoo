#!/bin/bash
#SBATCH --job-name=GZ2_ddpm
#SBATCH --output=GZ2_ddpm_%j.out
#SBATCH --error=GZ2_ddpm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=minerva
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gregory.sainton@obspm.fr

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

source /minerva/gsainton/miniconda3/etc/profile.d/conda.sh
conda activate LDM_model
export PYTHONPATH=/home/gsainton/01_CODES/DDPM_Galaxyzoo/src:$PYTHONPATH

export PYTHONPATH=${PROJECT_DIR}/src:$PYTHONPATH

echo "Project dir:   ${PROJECT_DIR}"
echo "PYTHONPATH:    ${PYTHONPATH}"
echo "Param file:    ${PARAM_FILE:-configs/param_GZ2.json}"

/minerva/gsainton/miniconda3/envs/LDM_model/bin/python \
    ${PROJECT_DIR}/scripts//home/gsainton/01_CODES/DDPM_Galaxyzoo/scripts/GZ2_DDPM_training_v5.py \
    --params ${PARAM_FILE:-configs/param_GZ2.json}

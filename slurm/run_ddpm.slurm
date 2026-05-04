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

export LD_PRELOAD=/minerva/gsainton/miniconda3/envs/ddpm_model/lib/libiomp5.so


source /minerva/gsainton/miniconda3/etc/profile.d/conda.sh
conda activate LDM_model
export PYTHONPATH=/obs/gsainton/DDPM_GalaxyZoo/src:$PYTHONPATH

export PYTHONPATH=${PROJECT_DIR}/src:$PYTHONPATH

echo "Project dir:   ${PROJECT_DIR}"
echo "PYTHONPATH:    ${PYTHONPATH}"
echo "Param file:    ${PARAM_FILE:-configs/param_GZ2.json}"

/minerva/gsainton/miniconda3/envs/LDM_model/bin/python ~/DDPM_GalaxyZoo/scripts/GZ2_DDPM_training_v5.py \
    -p ${PARAM_FILE:-configs/param_GZ2.json}

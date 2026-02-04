#!/bin/bash
#SBATCH --job-name=10_pr
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:15:00
#SBATCH --mem=16G
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

cd /n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/original_code

module purge
module load Mambaforge
module load cuda cudnn
mamba activate vlenv

python vllm_examples.py 

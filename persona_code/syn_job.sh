#!/bin/bash
#!/bin/bash
#SBATCH --job-name=syn_10
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:15:00
#SBATCH --mem=16G
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

cd /n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/persona_code

module purge
module load Mambaforge
module load cuda cudnn
mamba activate vlenv

template=math   # template can also be "npc", "knowledge" or "math". Feel free to try others; You can also add your customized data synthesis prompt in code/prompt_templates.py
sample_size=0  # Set sample_size=0 if you want to use the full version of 200k personas.
out_path=/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_personas/persona_code/examples/500_math_examples_4096_qwen38b.jsonl
model_path=Qwen/Qwen3-8B
#model_path=allenai/Olmo-3-7B-Think

# ensure that the necessary libraries such as transformers and vllm are installed or configured properly before running the following command.
PYTHONPATH=. python vllm_synthesize.py --model_path $model_path --template $template --sample_size $sample_size  --output_path $out_path
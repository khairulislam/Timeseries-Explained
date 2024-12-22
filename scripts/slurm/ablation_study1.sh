#!/usr/bin/env bash
#SBATCH --job-name="electricity_interpret"
#SBATCH --output=results/outputs/ablations_study_1.out
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# module load cuda-toolkit cudnn-8.9.5_cuda12.x miniforge
module load miniforge

conda activate ml

baseline_modes=('zero' 'random' 'aug' 'mean' 'gen')

for baseline_mode in ${baseline_modes[@]}
do 
echo $baseline_mode

python interpret.py \
  --task_name long_term_forecast \
  --explainers wtsr \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model iTransformer \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --result_path "results/ablation_study_1/$baseline_mode"\
  --baseline_mode $baseline_mode --disable_progress

done
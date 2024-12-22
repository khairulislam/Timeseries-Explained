#!/usr/bin/env bash
#SBATCH --job-name="timeLLM_train"
#SBATCH --output=outputs/timeLLM_train.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mail-type=begin,end
#---SBATCH --mail-user=email
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

python interpret_timeLLM.py \
  --task_name long_term_forecast \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation integrated_gradients gradient_shap winIT gatemask wtsr \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model TimeLLM \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 \
  --overwrite --disable_progress

python interpret_timeLLM.py \
  --task_name long_term_forecast \
  --explainers feature_ablation\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model TimeLLM \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1

python interpret_timeLLM.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation integrated_gradients gradient_shap winIT gatemask wtsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model DLinear --n_features 31  --seq_len 48 --disable_progress
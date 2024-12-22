#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=results/outputs/CALF_traffic_train.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

python run_CALF.py\
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model CALF \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --d_model 768

python run_CALF.py\
  --task_name long_term_forecast \
  --train \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model CALF \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --d_model 768

python run_CALF.py \
  --task_name classification \
  --data mimic \
  --train \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --model CALF --n_features 31\
  --seq_len 48 \
  --d_model 768 --task_loss ce
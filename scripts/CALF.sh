# #!/usr/bin/env bash
# #SBATCH --job-name="CALF"
# #SBATCH --output=results/outputs/calf.out
# #SBATCH --partition=gpu
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:v100:1
# #SBATCH --account=bii_dsc_community
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# module load cuda cudnn miniforge
# conda activate ml2
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/ml2/lib

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
  --itr_no 1 --d_model 768 --task_loss ce

python interpret_CALF.py\
  --task_name long_term_forecast \
  --explainers integrated_gradients gradient_shap\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model CALF \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --d_model 768
  
python interpret_CALF.py\
  --task_name long_term_forecast \
  --explainers tsr\
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model CALF \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --d_model 768

python interpret_CALF.py \
  --explainers integrated_gradients gradient_shap \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model CALF --n_features 31 --seq_len 48\
  --d_model 768 --overwrite --batch_size 16
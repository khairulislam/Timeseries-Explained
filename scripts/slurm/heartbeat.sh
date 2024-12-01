#!/usr/bin/env bash
#SBATCH --job-name="heatbeat"
#SBATCH --output=results/outputs/heatbeat.out
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mi3se@virginia.edu
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

python interpret.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation integrated_gradients gradient_shap winIT wtsr\
  --task_name classification \
  --data UEA \
  --root_path ./dataset/Heartbeat \
  --data_path Heartbeat \
  --metrics auc accuracy cross_entropy \
  --model iTransformer --n_features 61\
  --e_layers 3 \
  --seq_len 96\
  --label_len 48\
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 --disable_progress

# models=("DLinear" "MICN" "SegRNN" "iTransformer")

# for model in ${models[@]}
# do 
# python run.py \
#   --task_name classification \
#   --train \
#   --root_path ./dataset/Heartbeat \
#   --data_path Heartbeat\
#   --model $model \
#   --data UEA \
#   --e_layers 3 \
#   --seq_len 96\
#   --label_len 48\
#   --n_features 61\
#   --batch_size 16 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --learning_rate 0.001 \
#   --train_epochs 100 \
#   --patience 10 --disable_progress


# done

# for model in ${models[@]}
# do
# python interpret.py \
#   --explainers feature_ablation occlusion augmented_occlusion feature_permutation integrated_gradients gradient_shap winIT wtsr\
#   --task_name classification \
#   --data UEA \
#   --root_path ./dataset/Heartbeat \
#   --data_path Heartbeat \
#   --metrics auc accuracy cross_entropy \
#   --model $model --n_features 61\
#   --e_layers 3 \
#   --seq_len 96\
#   --label_len 48\
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 

# done
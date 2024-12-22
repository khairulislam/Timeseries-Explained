#!/usr/bin/env bash
#SBATCH --job-name="calf_interpret"
#---SBATCH --output=results/outputs/CALF_interpret.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

# Run this to submit this job
## sbatch --array=1-3 scripts/slurm/CALF_interpret.sh

source /etc/profile.d/modules.sh
source ~/.bashrc

module load miniforge

conda activate ml

explainers=("feature_ablation" "occlusion" "augmented_occlusion" "feature_permutation" "integrated_gradients" "gradient_shap" "dyna_mask" "extremal_mask" "winIT" "wtsr" "gatemask" "tsr")

function interpret {
  echo "Running $1-th job"

  python interpret_CALF.py\
    --task_name long_term_forecast \
    --explainers $explainers\
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model CALF \
    --features S \
    --seq_len 96 \
    --label_len 12 \
    --pred_len 24 \
    --n_features 1 --d_model 768\
    --disable_progress --itr_no $1

  python interpret_CALF.py\
    --task_name long_term_forecast \
    --explainers $explainers\
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model CALF \
    --features S \
    --seq_len 96 \
    --label_len 12 \
    --pred_len 24 \
    --n_features 1 --d_model 768\
    --disable_progress --itr_no $1

  python interpret_CALF.py \
    --explainers $explainers\
    --task_name classification \
    --data mimic \
    --root_path ./dataset/mimic_iii/ \
    --data_path mimic_iii.pkl \
    --metrics auc accuracy cross_entropy \
    --model CALF --n_features 31 --seq_len 48\
    --d_model 768 --overwrite \
    --disable_progress --itr_no $1
}

interpret $SLURM_ARRAY_TASK_ID
# max=3
# for (( i=1; i <= $max; ++i ))
# do
#     interpret $i
# done
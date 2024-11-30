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
  --n_features 1 --train_epochs 1 --itr_no 1 --d_model 768

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
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation integrated_gradients gradient_shap winIT gatemask wtsr\
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model CALF \
  --features S \
  --seq_len 96 \
  --label_len 12 \
  --pred_len 24 \
  --n_features 1 --d_model 768\
  --batch_size 16

python interpret_CALF.py \
  --explainers feature_ablation occlusion augmented_occlusion feature_permutation integrated_gradients gradient_shap winIT gatemask wtsr \
  --task_name classification \
  --data mimic \
  --root_path ./dataset/mimic_iii/ \
  --data_path mimic_iii.pkl \
  --metrics auc accuracy cross_entropy \
  --model CALF --n_features 31 --seq_len 48\
  --d_model 768 --overwrite --batch_size 16
python run.py \
  --task_name classification \
  --train \
  --root_path ./dataset/Heartbeat \
  --data_path Heartbeat\
  --model DLinear \
  --data UEA \
  --e_layers 3 \
  --seq_len 96\
  --label_len 48\
  --n_features 61\
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --task_name classification \
  --train \
  --root_path ./dataset/Heartbeat \
  --data_path Heartbeat\
  --model MICN \
  --data UEA \
  --e_layers 3 \
  --seq_len 96\
  --n_features 61\
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --task_name classification \
  --train \
  --root_path ./dataset/Heartbeat \
  --data_path Heartbeat\
  --model SegRNN \
  --data UEA \
  --e_layers 3 \
  --seq_len 96\
  --label_len 48\
  --n_features 61\
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --task_name classification \
  --train \
  --root_path ./dataset/Heartbeat \
  --data_path Heartbeat\
  --model iTransformer \
  --data UEA \
  --e_layers 3 \
  --seq_len 96\
  --label_len 48\
  --n_features 61\
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

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
  --top_k 3
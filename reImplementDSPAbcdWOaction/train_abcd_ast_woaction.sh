# set cuda device

# 1p
# export CUDA_VISIBLE_DEVICES=2,3
# python train.py --experiment_name abcdASTWAction1P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_waction_1p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_1p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_1p.json \
#   --text_column input \
#   --summary_column target \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --predict_with_generate \
#   --output_dir ./results/ \
#   --save_strategy epoch \
#   --source_prefix "Predict AST: " \
#   --max_source_length 1024 \
#   --max_target_length 256 \
#   --val_max_target_length 256 \
#   --learning_rate 5e-5 \
#   --warmup_steps 500 \
#   --use_ast_metrics \
#   --use_fast_tokenizer False \
#   --num_beams 4

# 10p
# export CUDA_VISIBLE_DEVICES=2,3
# python train.py --experiment_name abcdASTWAction10P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_waction_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_10p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_10p.json \
#   --text_column input \
#   --summary_column target \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --predict_with_generate \
#   --output_dir ./results/ \
#   --save_strategy epoch \
#   --source_prefix "Predict AST: " \
#   --max_source_length 1024 \
#   --max_target_length 256 \
#   --val_max_target_length 256 \
#   --learning_rate 5e-5 \
#   --warmup_steps 500 \
#   --use_ast_metrics \
#   --use_fast_tokenizer False \
#   --num_beams 4

# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWOAction20P"
# python train.py --experiment_name abcdASTWOAction20P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_20p.json \
#   --validation_file ./data/processed/dev_AST_abcd_20p.json \
#   --test_file ./data/processed/test_AST_abcd_20p.json \
#   --text_column input \
#   --summary_column target \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --predict_with_generate \
#   --output_dir ./results/ \
#   --save_strategy epoch \
#   --source_prefix "Predict AST: " \
#   --max_source_length 1024 \
#   --max_target_length 256 \
#   --val_max_target_length 256 \
#   --learning_rate 5e-5 \
#   --warmup_steps 500 \
#   --use_ast_metrics \
#   --use_fast_tokenizer False \
#   --num_beams 4


# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWOAction30P"
# python train.py --experiment_name abcdASTWOAction30P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_30p.json \
#   --validation_file ./data/processed/dev_AST_abcd_30p.json \
#   --test_file ./data/processed/test_AST_abcd_30p.json \
#   --text_column input \
#   --summary_column target \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --predict_with_generate \
#   --output_dir ./results/ \
#   --save_strategy epoch \
#   --source_prefix "Predict AST: " \
#   --max_source_length 1024 \
#   --max_target_length 256 \
#   --val_max_target_length 256 \
#   --learning_rate 5e-5 \
#   --warmup_steps 500 \
#   --use_ast_metrics \
#   --use_fast_tokenizer False \
#   --num_beams 4


# half
# export CUDA_VISIBLE_DEVICES=2,3
# python train.py --experiment_name abcdASTWActionHalf \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_waction_half.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_half.json \
#   --test_file ./data/processed/test_AST_abcd_waction_half.json \
#   --text_column input \
#   --summary_column target \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --predict_with_generate \
#   --output_dir ./results/ \
#   --save_strategy epoch \
#   --source_prefix "Predict AST: " \
#   --max_source_length 1024 \
#   --max_target_length 256 \
#   --val_max_target_length 256 \
#   --learning_rate 5e-5 \
#   --warmup_steps 500 \
#   --use_ast_metrics \
#   --use_fast_tokenizer False \
#   --num_beams 4

# all
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWOActionAll"
python train.py --experiment_name abcdASTWOActionAll \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_all.json \
  --validation_file ./data/processed/dev_AST_abcd_all.json \
  --test_file ./data/processed/test_AST_abcd_all.json \
  --text_column input \
  --summary_column target \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --predict_with_generate \
  --output_dir ./results/ \
  --save_strategy epoch \
  --source_prefix "Predict AST: " \
  --max_source_length 1024 \
  --max_target_length 256 \
  --val_max_target_length 256 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --use_ast_metrics \
  --use_fast_tokenizer False \
  --num_beams 4 \
# set cuda device
export CUDA_VISIBLE_DEVICES=6,7
echo "abcdASTWActionAll"
python train.py --experiment_name abcdASTWActionAll \
 --model_name_or_path results/abcdASTWActionAll_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_waction_full.json \
  --validation_file ./data/processed/dev_AST_abcd_waction_full.json \
  --test_file ./data/processed/dev_AST_abcd_waction_full.json \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 4


# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWActionHalf"
# python train.py --experiment_name abcdASTWActionHalf \
#  --model_name_or_path results/abcdASTWActionHalf_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWAction1P"
# python train.py --experiment_name abcdASTWAction1P \
#  --model_name_or_path results/abcdASTWAction1P_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWAction10P"
# python train.py --experiment_name abcdASTWAction10P \
#  --model_name_or_path results/abcdASTWAction10P_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWAction20P"
# python train.py --experiment_name abcdASTWAction20P \
#  --model_name_or_path results/abcdASTWAction20P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_waction_20p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_20p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_20p.json \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWAction30P"
# python train.py --experiment_name abcdASTWAction30P \
#  --model_name_or_path results/abcdASTWAction30P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_waction_30p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_30p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_30p.json \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4
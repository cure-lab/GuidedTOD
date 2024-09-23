# set cuda device

# 1p
# export CUDA_VISIBLE_DEVICES=0,1
# python train.py --experiment_name abcdMultiASTWAction1P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_waction_1p.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_waction_1p.json \
#   --test_file ./data/processed/test_multiAST_abcd_waction_1p.json \
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

# # delete the saved model
# rm -rf results/abcdMultiASTWAction1P_input_target_t5-small/checkpoint-*

# 10p
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdMultiASTWAction10P"
# python train.py --experiment_name abcdMultiASTWAction10P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_waction_10p.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_waction_10p.json \
#   --test_file ./data/processed/test_multiAST_abcd_waction_10p.json \
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

# # delete the saved model
# rm -rf results/abcdMultiASTWAction10P_input_target_t5-small/checkpoint-*

# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdMultiASTWAction20P"
# python train.py --experiment_name abcdMultiASTWAction20P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_waction_20p.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_waction_20p.json \
#   --test_file ./data/processed/test_multiAST_abcd_waction_20p.json \
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

# # delete the saved model
# rm -rf results/abcdMultiASTWAction20P_input_target_t5-small/checkpoint-*

# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdMultiASTWAction30P"
# python train.py --experiment_name abcdMultiASTWAction30P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_waction_30p.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_waction_30p.json \
#   --test_file ./data/processed/test_multiAST_abcd_waction_30p.json \
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

# # delete the saved model
# rm -rf results/abcdMultiASTWAction30P_input_target_t5-small/checkpoint-*

# half
# export CUDA_VISIBLE_DEVICES=0,1
# python train.py --experiment_name abcdMultiASTWActionHalf \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_waction_half.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_waction_half.json \
#   --test_file ./data/processed/test_multiAST_abcd_waction_half.json \
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

# # delete the saved model
# rm -rf results/abcdMultiASTWActionHalf_input_target_t5-small/checkpoint-*

# all
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdMultiASTWActionAll30Epoch"
python train.py --experiment_name abcdMultiASTWActionAll30Epoch \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 30 \
  --train_file ./data/processed/train_multiAST_abcd_waction_all.json \
  --validation_file ./data/processed/dev_multiAST_abcd_waction_all.json \
  --test_file ./data/processed/test_multiAST_abcd_waction_all.json \
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
  --num_beams 1

  # delete the saved model
rm -rf results/abcdMultiASTWActionAll30Epoch_input_target_t5-small/checkpoint-*
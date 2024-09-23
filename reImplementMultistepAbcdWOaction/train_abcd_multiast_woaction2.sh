# set cuda device

# 1p
# export CUDA_VISIBLE_DEVICES=3,4
# echo "abcdMultiASTWOAction1P"
# python train.py --experiment_name abcdMultiASTWOAction1P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_woaction_1p.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_woaction_1p.json \
#   --test_file ./data/processed/test_multiAST_abcd_woaction_1p.json \
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
# rm -rf results/abcdMultiASTWOAction1P_input_target_t5-small/checkpoint-*

# 10p
export CUDA_VISIBLE_DEVICES=6,7
echo "abcdMultiASTWOAction10P"
python train.py --experiment_name abcdMultiASTWOAction10P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_abcd_woaction_10p.json \
  --validation_file ./data/processed/dev_multiAST_abcd_woaction_10p.json \
  --test_file ./data/processed/test_multiAST_abcd_woaction_10p.json \
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
  --num_beams 4

# delete the saved model
rm -rf results/abcdMultiASTWOAction10P_input_target_t5-small/checkpoint-*

export CUDA_VISIBLE_DEVICES=6,7
echo "abcdMultiASTWOAction20P"
python train.py --experiment_name abcdMultiASTWOAction20P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_abcd_woaction_20p.json \
  --validation_file ./data/processed/dev_multiAST_abcd_woaction_20p.json \
  --test_file ./data/processed/test_multiAST_abcd_woaction_20p.json \
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
  --num_beams 4

# delete the saved model
rm -rf results/abcdMultiASTWOAction20P_input_target_t5-small/checkpoint-*

export CUDA_VISIBLE_DEVICES=6,7
echo "abcdMultiASTWOAction30P"
python train.py --experiment_name abcdMultiASTWOAction30P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_abcd_woaction_30p.json \
  --validation_file ./data/processed/dev_multiAST_abcd_woaction_30p.json \
  --test_file ./data/processed/test_multiAST_abcd_woaction_30p.json \
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
  --num_beams 4

# delete the saved model
rm -rf results/abcdMultiASTWOAction30P_input_target_t5-small/checkpoint-*

# half
export CUDA_VISIBLE_DEVICES=6,7
echo "abcdMultiASTWOActionHalf"
python train.py --experiment_name abcdMultiASTWOActionHalf \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_abcd_woaction_half.json \
  --validation_file ./data/processed/dev_multiAST_abcd_woaction_half.json \
  --test_file ./data/processed/test_multiAST_abcd_woaction_half.json \
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
  --num_beams 4

# delete the saved model
rm -rf results/abcdMultiASTWOActionHalf_input_target_t5-small/checkpoint-*

# all
# export CUDA_VISIBLE_DEVICES=3,4
# echo "abcdMultiASTWOActionAll"
# python train.py --experiment_name abcdMultiASTWOActionAll \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_abcd_woaction_all.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_woaction_all.json \
#   --test_file ./data/processed/test_multiAST_abcd_woaction_all.json \
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

#   # delete the saved model
# rm -rf results/abcdMultiASTWOActionAll_input_target_t5-small/checkpoint-*
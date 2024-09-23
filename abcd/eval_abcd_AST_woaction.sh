# set cuda device
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWOActionAll"
python train.py --experiment_name abcdASTWOActionAll \
 --model_name_or_path results/abcdASTWOActionAll_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd-full.json \
  --validation_file ./data/processed/dev_AST_abcd-full.json \
  --test_file ./data/processed/test_AST_abcd-full.json \
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


# export CUDA_VISIBLE_DEVICES=2,3
# echo "abcdASTWOActionHalf"
# python train.py --experiment_name abcdASTWOActionHalf \
#  --model_name_or_path results/abcdASTWOActionHalf_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_half.json \
#   --validation_file ./data/processed/dev_AST_abcd_half.json \
#   --test_file ./data/processed/test_AST_abcd_half.json \
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


# export CUDA_VISIBLE_DEVICES=2,3
# echo "abcdASTWOAction1P"
# python train.py --experiment_name abcdASTWOAction1P \
#  --model_name_or_path results/abcdASTWOAction1P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_1p.json \
#   --validation_file ./data/processed/dev_AST_abcd_1p.json \
#   --test_file ./data/processed/test_AST_abcd_1p.json \
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


# # export CUDA_VISIBLE_DEVICES=0,1
# # echo "abcdASTWOAction30P"
# # python train.py --experiment_name abcdASTWOAction30P \
# #  --model_name_or_path results/abcdASTWOAction30P_input_target_t5-small \
# #   --do_predict \
# #   --train_file ./data/processed/train_AST_abcd_30p.json \
# #   --validation_file ./data/processed/dev_AST_abcd_30p.json \
# #   --test_file ./data/processed/test_AST_abcd_30p.json \
# #   --text_column input \
# #   --summary_column target \
# #   --per_device_train_batch_size 32 \
# #   --per_device_eval_batch_size 32 \
# #   --predict_with_generate \
# #   --output_dir ./results/ \
# #   --save_strategy epoch \
# #   --source_prefix "Predict AST: " \
# #   --max_source_length 1024 \
# #   --max_target_length 256 \
# #   --val_max_target_length 256 \
# #   --learning_rate 5e-5 \
# #   --warmup_steps 500 \
# #   --use_fast_tokenizer False \
# #   --use_ast_metrics \
# #   --num_beams 4

# export CUDA_VISIBLE_DEVICES=2,3
# echo "abcdASTWOAction10P"
# python train.py --experiment_name abcdASTWOAction10P \
#  --model_name_or_path results/abcdASTWOAction10P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_10p.json \
#   --test_file ./data/processed/test_AST_abcd_10p.json \
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

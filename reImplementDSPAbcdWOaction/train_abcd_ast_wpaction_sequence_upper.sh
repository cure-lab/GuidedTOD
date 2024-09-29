# set cuda device

# 1p
# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWPAction1P"
# python train.py --experiment_name abcdASTWPAction1P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_1p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_1p.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_1p.json \
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
# rm -rf results/abcdASTWPAction1P_input_target_t5-small/checkpoint-*

# # 10p
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWPActionSequence10PUpper"
python train.py --experiment_name abcdASTWPActionSequence10PUpper \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wpaction_sequence_upper_10p.json \
  --validation_file ./data/processed/dev_AST_abcd_wpaction_sequence_upper_10p.json \
  --test_file ./data/processed/test_AST_abcd_wpaction_sequence_upper_10p.json \
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
rm -rf results/abcdASTWPActionSequence10PUpper_input_target_t5-small/checkpoint-*

# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWPAction20P"
# python train.py --experiment_name abcdASTWPAction20P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_20p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_20p.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_20p.json \
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
# rm -rf results/abcdASTWPAction20P_input_target_t5-small/checkpoint-*


export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWPActionSequence30PUpper"
python train.py --experiment_name abcdASTWPActionSequence30PUpper \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wpaction_sequence_upper_30p.json \
  --validation_file ./data/processed/dev_AST_abcd_wpaction_sequence_upper_30p.json \
  --test_file ./data/processed/test_AST_abcd_wpaction_sequence_upper_30p.json \
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
rm -rf results/abcdASTWPActionSequence30PUpper_input_target_t5-small/checkpoint-*

# # half
# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWPActionHalf"
# python train.py --experiment_name abcdASTWPActionHalf \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_half.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_half.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_half.json \
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
# rm -rf results/abcdASTWPActionHalf_input_target_t5-small/checkpoint-*

# # all
# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWPActionAll"
# python train.py --experiment_name abcdASTWPActionAll \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_full.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_full.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_full.json \
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
# rm -rf results/abcdASTWPActionAll_input_target_t5-small/checkpoint-*
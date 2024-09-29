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
# echo "multiwozASTWOAction10P"
# python train.py --experiment_name multiwozASTWOAction10P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_multiwoz_woaction_10p.json \
#   --validation_file ./data/processed/validation_AST_multiwoz_woaction_10p.json \
#   --test_file ./data/processed/test_AST_multiwoz_woaction_10p.json \
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
# rm -rf results/multiwozASTWOAction10P_input_target_t5-small/checkpoint-*

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


# all
export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozASTWOActionAll"
python train.py --experiment_name multiwozASTWOActionAll \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_woaction_all.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_all.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_all.json \
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
rm -rf results/multiwozASTWOActionAll_input_target_t5-small/checkpoint-*

# 30P
export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozASTWOAction30P"
python train.py --experiment_name multiwozASTWOAction30P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_woaction_30p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_30p.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_30p.json \
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
rm -rf results/multiwozASTWOAction30P_input_target_t5-small/checkpoint-*

# Half
export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozASTWOActionHalf"
python train.py --experiment_name multiwozASTWOActionHalf \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_woaction_half.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_half.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_half.json \
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
rm -rf results/multiwozASTWOActionHalf_input_target_t5-small/checkpoint-*
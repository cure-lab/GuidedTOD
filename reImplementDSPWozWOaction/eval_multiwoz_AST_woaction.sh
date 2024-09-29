# set cuda device
export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozASTWOAction10P"
python train.py --experiment_name multiwozASTWOAction10P \
 --model_name_or_path results/multiwozASTWOAction10P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_multiwoz_woaction_10p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_10p.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_10p.json \
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



# export CUDA_VISIBLE_DEVICES=0,1
# echo "multiwozASTWOAction10P"
# python train.py --experiment_name multiwozASTWOAction10P \
#  --model_name_or_path results/multiwozASTWOAction10P_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozASTWOActionHalf"
python train.py --experiment_name multiwozASTWOActionHalf \
 --model_name_or_path results/multiwozASTWOActionHalf_input_target_t5-small \
  --do_predict \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 4


# export CUDA_VISIBLE_DEVICES=0,1
# echo "multiwozASTWOActionAll"
# python train.py --experiment_name multiwozASTWOActionAll \
#  --model_name_or_path results/multiwozASTWOActionAll_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_multiwoz_woaction_all.json \
#   --validation_file ./data/processed/validation_AST_multiwoz_woaction_all.json \
#   --test_file ./data/processed/test_AST_multiwoz_woaction_all.json \
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


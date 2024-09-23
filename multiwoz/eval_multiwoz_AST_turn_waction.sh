export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozASTTurnWAction10P"
python train.py --experiment_name multiwozASTTurnWAction10P \
 --model_name_or_path results/multiwozASTTurnWAction10P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_multiwoz_turn_waction_10p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_turn_waction_10p.json \
  --test_file ./data/processed/test_AST_multiwoz_turn_waction_10p.json \
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
# echo "multiwozASTTurnWAction30P"
# python train.py --experiment_name multiwozASTTurnWAction30P \
#  --model_name_or_path results/multiwozASTTurnWAction30P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_multiwoz_turn_waction_30p.json \
#   --validation_file ./data/processed/validation_AST_multiwoz_turn_waction_30p.json \
#   --test_file ./data/processed/test_AST_multiwoz_turn_waction_30p.json \
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
# echo "multiwozASTTurnWActionHalf"
# python train.py --experiment_name multiwozASTTurnWActionHalf \
#  --model_name_or_path results/multiwozASTTurnWActionHalf_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_multiwoz_turn_waction_half.json \
#   --validation_file ./data/processed/validation_AST_multiwoz_turn_waction_half.json \
#   --test_file ./data/processed/test_AST_multiwoz_turn_waction_half.json \
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
# echo "multiwozASTTurnWActionAll"
# python train.py --experiment_name multiwozASTTurnWActionAll \
#  --model_name_or_path results/multiwozASTTurnWActionAll_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_multiwoz_turn_waction_all.json \
#   --validation_file ./data/processed/validation_AST_multiwoz_turn_waction_all.json \
#   --test_file ./data/processed/test_AST_multiwoz_turn_waction_all.json \
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
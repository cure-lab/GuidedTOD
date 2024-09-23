# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdMultiASTWOAction10P"
# python train.py --experiment_name abcdMultiASTWOAction10P \
#  --model_name_or_path results/abcdMultiASTWOAction10P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_abcd_woaction_10p.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_woaction_10p.json \
#   --test_file ./data/processed/test_multiAST_abcd_woaction_10p.json \
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
#   --num_beams 1


export CUDA_VISIBLE_DEVICES=0,1
echo "abcdMultiASTWOAction30P"
python train.py --experiment_name abcdMultiASTWOAction30P \
 --model_name_or_path results/abcdMultiASTWOAction30P_input_target_t5-small \
  --do_predict \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 1


# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdMultiASTWOActionHalf"
# python train.py --experiment_name abcdMultiASTWOActionHalf \
#  --model_name_or_path results/abcdMultiASTWOActionHalf_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_abcd_woaction_half.json \
#   --validation_file ./data/processed/dev_multiAST_abcd_woaction_half.json \
#   --test_file ./data/processed/test_multiAST_abcd_woaction_half.json \
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
#   --num_beams 1

# export CUDA_VISIBLE_DEVICES=1,2
# echo "abcdMultiASTWOActionAll"
# python train.py --experiment_name abcdMultiASTWOActionAll \
#  --model_name_or_path results/abcdMultiASTWOActionAll_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 1
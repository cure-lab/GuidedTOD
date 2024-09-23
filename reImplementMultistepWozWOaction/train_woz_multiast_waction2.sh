# set cuda device

# all
# export CUDA_VISIBLE_DEVICES=0,1
# echo "multiwozMultiASTWActionAll"
# python train.py --experiment_name multiwozMultiASTWActionAll \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_all.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_all.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_all.json \
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
#   --num_beams 1

#   # delete the saved model
# rm -rf results/multiwozMultiASTWActionAll_input_target_t5-small/checkpoint-*

export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozMultiASTWAction30P"
python train.py --experiment_name multiwozMultiASTWAction30P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_waction_30p.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_waction_30p.json \
  --test_file ./data/processed/test_multiAST_multiwoz_waction_30p.json \
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
rm -rf results/multiwozMultiASTWAction30P_input_target_t5-small/checkpoint-*


# 10p
export CUDA_VISIBLE_DEVICES=0,1
echo "multiwozMultiASTWAction10P"
python train.py --experiment_name multiwozMultiASTWAction10P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_waction_10p.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_waction_10p.json \
  --test_file ./data/processed/test_multiAST_multiwoz_waction_10p.json \
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
rm -rf results/multiwozMultiASTWAction10P_input_target_t5-small/checkpoint-*


# half
export CUDA_VISIBLE_DEVICES=0,1
python train.py --experiment_name multiwozMultiASTWActionHalf \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_waction_half.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_waction_half.json \
  --test_file ./data/processed/test_multiAST_multiwoz_waction_half.json \
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
rm -rf results/multiwozMultiASTWActionHalf_input_target_t5-small/checkpoint-*
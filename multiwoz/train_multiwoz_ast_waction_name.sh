# set cuda device

# 10p
export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWActionName10P"
python train.py --experiment_name multiwozASTWActionName10P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_waction_name_10p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_waction_name_10p.json \
  --test_file ./data/processed/test_AST_multiwoz_waction_name_10p.json \
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
rm -rf results/multiwozASTWActionName10P_input_target_t5-small/checkpoint-*

# 30P
export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWActionName30P"
python train.py --experiment_name multiwozASTWActionName30P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_waction_name_30p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_waction_name_30p.json \
  --test_file ./data/processed/test_AST_multiwoz_waction_name_30p.json \
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
rm -rf results/multiwozASTWActionName30P_input_target_t5-small/checkpoint-*

# Half
export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWActionNameHalf"
python train.py --experiment_name multiwozASTWActionNameHalf \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_waction_name_half.json \
  --validation_file ./data/processed/validation_AST_multiwoz_waction_name_half.json \
  --test_file ./data/processed/test_AST_multiwoz_waction_name_half.json \
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
rm -rf results/multiwozASTWActionNameHalf_input_target_t5-small/checkpoint-*

# all
export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWActionNameAll"
python train.py --experiment_name multiwozASTWActionNameAll \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_multiwoz_waction_name_all.json \
  --validation_file ./data/processed/validation_AST_multiwoz_waction_name_all.json \
  --test_file ./data/processed/test_AST_multiwoz_waction_name_all.json \
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
rm -rf results/multiwozASTWActionNameAll_input_target_t5-small/checkpoint-*

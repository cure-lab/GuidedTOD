# set cuda device


# # 10p
export CUDA_VISIBLE_DEVICES=1,2
echo "multiwozMultiASTWOAction10P100"
python train.py --experiment_name multiwozMultiASTWOAction10P100 \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_woaction_10p.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_woaction_10p.json \
  --test_file ./data/processed/test_multiAST_multiwoz_woaction_10p.json \
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
rm -rf results/multiwozMultiASTWOAction10P100_input_target_t5-small/checkpoint-*



export CUDA_VISIBLE_DEVICES=1,2
echo "multiwozMultiASTWOAction30P100"
python train.py --experiment_name multiwozMultiASTWOAction30P100 \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_woaction_30p.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_woaction_30p.json \
  --test_file ./data/processed/test_multiAST_multiwoz_woaction_30p.json \
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
rm -rf results/multiwozMultiASTWOAction30P100_input_target_t5-small/checkpoint-*


# # half
export CUDA_VISIBLE_DEVICES=1,2
echo "multiwozMultiASTWOActionHalf100"
python train.py --experiment_name multiwozMultiASTWOActionHalf100 \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_woaction_half.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_woaction_half.json \
  --test_file ./data/processed/test_multiAST_multiwoz_woaction_half.json \
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
rm -rf results/multiwozMultiASTWOActionHalf100_input_target_t5-small/checkpoint-*

# all
export CUDA_VISIBLE_DEVICES=1,2
echo "multiwozMultiASTWOActionAll100"
python train.py --experiment_name multiwozMultiASTWOActionAll100 \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_multiAST_multiwoz_woaction_all.json \
  --validation_file ./data/processed/validation_multiAST_multiwoz_woaction_all.json \
  --test_file ./data/processed/test_multiAST_multiwoz_woaction_all.json \
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
rm -rf results/multiwozMultiASTWOActionAll100_input_target_t5-small/checkpoint-*

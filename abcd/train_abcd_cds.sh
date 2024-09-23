# set cuda device
export CUDA_VISIBLE_DEVICES=6,7
python train.py --experiment_name multiwozAST \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_CDS_abcd.json \
  --validation_file ./data/processed/dev_CDS_abcd.json \
  --test_file ./data/processed/test_CDS_abcd.json \
  --text_column input \
  --summary_column target \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
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
  --use_fast_tokenizer False
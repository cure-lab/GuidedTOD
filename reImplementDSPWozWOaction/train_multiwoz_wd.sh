# set cuda device
export CUDA_VISIBLE_DEVICES=4,5
python train.py --experiment_name multiwozWD \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_workflow_discovery_multiwoz.json \
  --validation_file ./data/processed/validation_workflow_discovery_multiwoz.json \
  --test_file ./data/processed/test_workflow_discovery_multiwoz.json \
  --text_column input \
  --summary_column target \
  --per_device_train_batch_size 48 \
  --per_device_eval_batch_size 48 \
  --predict_with_generate \
  --output_dir ./results/ \
  --save_strategy epoch \
  --source_prefix "Extract workflow : " \
  --max_source_length 1024 \
  --max_target_length 256 \
  --val_max_target_length 256 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --use_fast_tokenizer False
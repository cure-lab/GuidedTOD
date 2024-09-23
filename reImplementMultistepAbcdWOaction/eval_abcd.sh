# set cuda device
export CUDA_VISIBLE_DEVICES=4,5
python train.py --experiment_name my_wd_experiment \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file ./data/processed/train_workflow_discovery_abcd.json \
  --validation_file ./data/processed/dev_workflow_discovery_abcd.json \
  --test_file ./data/processed/dev_workflow_discovery_abcd.json \
  --text_column input \
  --summary_column target \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --predict_with_generate \
  --output_dir ./results/ \
  --save_strategy epoch \
  --source_prefix "Extract workflow : " \
  --max_source_length 1024 \
  --max_target_length 256 \
  --val_max_target_length 256 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --ignore_pad_token_for_loss \
  --use_fast_tokenizer False \
  --resume_from_checkpoint results/my_wd_experiment_input_target_t5-small/checkpoint-6237
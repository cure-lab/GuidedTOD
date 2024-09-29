# set cuda device
export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWOActionNameFlow10P"
python train.py --experiment_name multiwozASTWOActionNameFlow10P \
 --model_name_or_path results/multiwozASTWOActionNameFlow10P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_multiwoz_woaction_name_flow_10p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_name_flow_10p.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_name_flow_10p.json \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 4



export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWOActionNameFlow30P"
python train.py --experiment_name multiwozASTWOActionNameFlow30P \
 --model_name_or_path results/multiwozASTWOActionNameFlow30P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_multiwoz_woaction_name_flow_30p.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_name_flow_30p.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_name_flow_30p.json \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 4


export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWOActionNameFlowHalf"
python train.py --experiment_name multiwozASTWOActionNameFlowHalf \
 --model_name_or_path results/multiwozASTWOActionNameFlowHalf_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_multiwoz_woaction_name_flow_half.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_name_flow_half.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_name_flow_half.json \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 4


export CUDA_VISIBLE_DEVICES=6,7
echo "multiwozASTWOActionNameFlowAll"
python train.py --experiment_name multiwozASTWOActionNameFlowAll \
 --model_name_or_path results/multiwozASTWOActionNameFlowAll_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_multiwoz_woaction_name_flow_all.json \
  --validation_file ./data/processed/validation_AST_multiwoz_woaction_name_flow_all.json \
  --test_file ./data/processed/test_AST_multiwoz_woaction_name_flow_all.json \
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
  --use_fast_tokenizer False \
  --use_ast_metrics \
  --num_beams 4


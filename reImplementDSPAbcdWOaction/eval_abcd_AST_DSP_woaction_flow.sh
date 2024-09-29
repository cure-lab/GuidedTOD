# set cuda device
export CUDA_VISIBLE_DEVICES=2,3
echo "abcdASTWOActionFlowAll"
python train.py --experiment_name abcdASTWOActionFlowAll \
 --model_name_or_path results/abcdASTWOActionFlowAll_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_woaction_flow_all.json \
  --validation_file ./data/processed/dev_AST_abcd_woaction_flow_all.json \
  --test_file ./data/processed/test_AST_abcd_woaction_flow_all.json \
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


export CUDA_VISIBLE_DEVICES=2,3
echo "abcdASTWOActionFlowHalf"
python train.py --experiment_name abcdASTWOActionFlowHalf \
 --model_name_or_path results/abcdASTWOActionFlowHalf_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_woaction_flow_half.json \
  --validation_file ./data/processed/dev_AST_abcd_woaction_flow_half.json \
  --test_file ./data/processed/test_AST_abcd_woaction_flow_half.json \
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


export CUDA_VISIBLE_DEVICES=2,3
echo "abcdASTWOActionFlow30P"
python train.py --experiment_name abcdASTWOActionFlow30P \
 --model_name_or_path results/abcdASTWOActionFlow30P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_woaction_flow_30p.json \
  --validation_file ./data/processed/dev_AST_abcd_woaction_flow_30p.json \
  --test_file ./data/processed/test_AST_abcd_woaction_flow_30p.json \
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

export CUDA_VISIBLE_DEVICES=2,3
echo "abcdASTWOActionFlow10P"
python train.py --experiment_name abcdASTWOActionFlow10P \
 --model_name_or_path results/abcdASTWOActionFlow10P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_woaction_flow_10p.json \
  --validation_file ./data/processed/dev_AST_abcd_woaction_flow_10p.json \
  --test_file ./data/processed/test_AST_abcd_woaction_flow_10p.json \
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

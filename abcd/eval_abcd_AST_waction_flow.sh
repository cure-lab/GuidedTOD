# set cuda device
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWActionFlowAll"
# python train.py --experiment_name abcdASTWActionFlowAll \
#  --model_name_or_path results/abcdASTWActionFlowAll_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_waction_flow_all.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_flow_all.json \
#   --test_file ./data/processed/test_AST_abcd_waction_flow_all.json \
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


export CUDA_VISIBLE_DEVICES=6,7
echo "abcdASTWActionFlowHalf"
python train.py --experiment_name abcdASTWActionFlowHalf \
 --model_name_or_path results/abcdASTWActionFlowHalf_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_waction_flow_half.json \
  --validation_file ./data/processed/dev_AST_abcd_waction_flow_half.json \
  --test_file ./data/processed/test_AST_abcd_waction_flow_half.json \
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
# echo "abcdASTWAction1P"
# python train.py --experiment_name abcdASTWAction1P \
#  --model_name_or_path results/abcdASTWAction1P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_waction_1p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_1p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_1p.json \
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
# echo "abcdASTWActionFlow10P"
# python train.py --experiment_name abcdASTWActionFlow10P \
#  --model_name_or_path results/abcdASTWActionFlow10P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_waction_flow_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_flow_10p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_flow_10p.json \
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


# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWAction20P"
# python train.py --experiment_name abcdASTWAction20P \
#  --model_name_or_path results/abcdASTWAction20P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_waction_20p.json \
#   --validation_file ./data/processed/dev_AST_abcd_waction_20p.json \
#   --test_file ./data/processed/test_AST_abcd_waction_20p.json \
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


export CUDA_VISIBLE_DEVICES=6,7
echo "abcdASTWActionFlow30P"
python train.py --experiment_name abcdASTWActionFlow30P \
 --model_name_or_path results/abcdASTWActionFlow30P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_waction_flow_30p.json \
  --validation_file ./data/processed/dev_AST_abcd_waction_flow_30p.json \
  --test_file ./data/processed/test_AST_abcd_waction_flow_30p.json \
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
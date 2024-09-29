# set cuda device

# 1p
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWMostPActionChain1P"
# python train.py --experiment_name abcdASTWMostPActionChain1P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_1p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_1p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_1p.json \
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
#   --num_beams 4

# # delete the saved model
# rm -rf results/abcdASTWMostPActionChain1P_input_target_t5-small/checkpoint-*

# 10p
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWMostPActionChainFlow10P"
# python train.py --experiment_name abcdASTWMostPActionChainFlow10P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_flow_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_flow_10p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_flow_10p.json \
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
#   --num_beams 4

# # delete the saved model
# rm -rf results/abcdASTWMostPActionChainFlow10P_input_target_t5-small/checkpoint-*

# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWMostPActionChain20P"
# python train.py --experiment_name abcdASTWMostPActionChain20P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_20p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_20p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_20p.json \
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
#   --num_beams 4

# # delete the saved model
# rm -rf results/abcdASTWMostPActionChain20P_input_target_t5-small/checkpoint-*


export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWMostPActionChainFlow30P"
python train.py --experiment_name abcdASTWMostPActionChainFlow30P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wmostpaction_flow_30p.json \
  --validation_file ./data/processed/dev_AST_abcd_wmostpaction_flow_30p.json \
  --test_file ./data/processed/test_AST_abcd_wmostpaction_flow_30p.json \
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
rm -rf results/abcdASTWMostPActionChainFlow30P_input_target_t5-small/checkpoint-*

# half
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWMostPActionChainFlowHalf"
python train.py --experiment_name abcdASTWMostPActionChainFlowHalf \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wmostpaction_flow_half.json \
  --validation_file ./data/processed/dev_AST_abcd_wmostpaction_flow_half.json \
  --test_file ./data/processed/test_AST_abcd_wmostpaction_flow_half.json \
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
rm -rf results/abcdASTWMostPActionChainFlowHalf_input_target_t5-small/checkpoint-*

# all
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWMostPActionChainFlowAll"
# python train.py --experiment_name abcdASTWMostPActionChainFlowAll \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_flow_all.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_flow_all.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_flow_all.json \
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
#   --num_beams 4

# # delete the saved model
# rm -rf results/abcdASTWMostPActionChainFlowAll_input_target_t5-small/checkpoint-*
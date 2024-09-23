# set cuda device

# 1p
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWPActionChain1P"
python train.py --experiment_name abcdASTWPActionChain1P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wpaction_chain_1p.json \
  --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_1p.json \
  --test_file ./data/processed/test_AST_abcd_wpaction_chain_1p.json \
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
rm -rf results/abcdASTWPActionChain1P_input_target_t5-small/checkpoint-*

# 10p
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWPActionChain10P"
python train.py --experiment_name abcdASTWPActionChain10P \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wpaction_chain_10p.json \
  --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_10p.json \
  --test_file ./data/processed/test_AST_abcd_wpaction_chain_10p.json \
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
rm -rf results/abcdASTWPActionChain10P_input_target_t5-small/checkpoint-*

# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWPActionChain20P"
# python train.py --experiment_name abcdASTWPActionChain20P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_chain_20p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_20p.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_chain_20p.json \
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
# rm -rf results/abcdASTWPActionChain20P_input_target_t5-small/checkpoint-*


# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWPActionChain30P"
# python train.py --experiment_name abcdASTWPActionChain30P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_chain_30p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_30p.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_chain_30p.json \
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
# rm -rf results/abcdASTWPActionChain30P_input_target_t5-small/checkpoint-*

# # half
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWPActionChainHalf"
# python train.py --experiment_name abcdASTWPActionChainHalf \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wpaction_chain_half.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_half.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_chain_half.json \
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
# rm -rf results/abcdASTWPActionChainHalf_input_target_t5-small/checkpoint-*

# all
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWPActionChainAll"
python train.py --experiment_name abcdASTWPActionChainAll \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wpaction_chain_all.json \
  --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_all.json \
  --test_file ./data/processed/test_AST_abcd_wpaction_chain_all.json \
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
rm -rf results/abcdASTWPActionChainAll_input_target_t5-small/checkpoint-*
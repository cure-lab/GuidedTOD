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
# echo "abcdASTWMostPActionNoinitChain10P"
# python train.py --experiment_name abcdASTWMostPActionNoinitChain10P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_noinit_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_noinit_10p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_noinit_10p.json \
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
# rm -rf results/abcdASTWMostPActionNoinitChain10P_input_target_t5-small/checkpoint-*

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


# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWMostPActionNoinitChain30P"
# python train.py --experiment_name abcdASTWMostPActionNoinitChain30P \
#  --model_name_or_path t5-small \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --num_train_epochs 100 \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_noinit_30p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_noinit_30p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_noinit_30p.json \
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
# rm -rf results/abcdASTWMostPActionNoinitChain30P_input_target_t5-small/checkpoint-*

# # half
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWMostPActionNoinitChainHalf"
python train.py --experiment_name abcdASTWMostPActionNoinitChainHalf \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_noinit_half.json \
  --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_noinit_half.json \
  --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_noinit_half.json \
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
rm -rf results/abcdASTWMostPActionNoinitChainHalf_input_target_t5-small/checkpoint-*

# all
export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWMostPActionNoinitChainAll"
python train.py --experiment_name abcdASTWMostPActionNoinitChainAll \
 --model_name_or_path t5-small \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_noinit_all.json \
  --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_noinit_all.json \
  --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_noinit_all.json \
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
rm -rf results/abcdASTWMostPActionNoinitChainAll_input_target_t5-small/checkpoint-*
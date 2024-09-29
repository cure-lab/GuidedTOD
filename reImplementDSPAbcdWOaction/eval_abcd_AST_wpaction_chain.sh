# set cuda device
# export CUDA_VISIBLE_DEVICES=6,7
# echo "abcdASTWPActionChainAll"
# python train.py --experiment_name abcdASTWPActionChainAll \
#  --model_name_or_path results/abcdASTWPActionChainAll_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wpaction_chain_all.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_all.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_chain_all.json \
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
# echo "abcdASTWPActionChainHalf"
# python train.py --experiment_name abcdASTWPActionChainHalf \
#  --model_name_or_path results/abcdASTWPActionChainHalf_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWPActionChain1P"
# python train.py --experiment_name abcdASTWPActionChain1P \
#  --model_name_or_path results/abcdASTWPActionChain1P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wpaction_chain_1p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_1p.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_chain_1p.json \
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
# echo "abcdASTWPActionChain10P"
# python train.py --experiment_name abcdASTWPActionChain10P \
#  --model_name_or_path results/abcdASTWPActionChain10P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wpaction_chain_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_10p.json \
#   --test_file ./data/processed/test_AST_abcd_wpaction_chain_10p.json \
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
# echo "abcdASTWPActionChain20P"
# python train.py --experiment_name abcdASTWPActionChain20P \
#  --model_name_or_path results/abcdASTWPActionChain20P_input_target_t5-small \
#   --do_predict \
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
#   --use_fast_tokenizer False \
#   --use_ast_metrics \
#   --num_beams 4


export CUDA_VISIBLE_DEVICES=6,7
echo "abcdASTWPActionChain30P"
python train.py --experiment_name abcdASTWPActionChain30P \
 --model_name_or_path results/abcdASTWPActionChain30P_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_wpaction_chain_30p.json \
  --validation_file ./data/processed/dev_AST_abcd_wpaction_chain_30p.json \
  --test_file ./data/processed/test_AST_abcd_wpaction_chain_30p.json \
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
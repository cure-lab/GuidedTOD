# set cuda device
# export CUDA_VISIBLE_DEVICES=0,1
# echo "abcdASTWMostPActionChainProbAll30Epoch"
# python train.py --experiment_name abcdASTWMostPActionChainProbAll30Epoch \
#  --model_name_or_path results/abcdASTWMostPActionChainProbAll30Epoch_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_prob_all.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_prob_all.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_prob_all.json \
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

export CUDA_VISIBLE_DEVICES=0,1
echo "abcdASTWMostPActionChainProbAll"
python train.py --experiment_name abcdASTWMostPActionChainProbAll \
 --model_name_or_path results/abcdASTWMostPActionChainProbAll_input_target_t5-small \
  --do_predict \
  --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_prob_all.json \
  --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_prob_all.json \
  --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_prob_all.json \
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
# echo "abcdASTWMostPActionChainProbHalf"
# python train.py --experiment_name abcdASTWMostPActionChainProbHalf \
#  --model_name_or_path results/abcdASTWMostPActionChainProbHalf_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_prob_half.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_prob_half.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_prob_half.json \
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
# echo "abcdASTWMostPActionChainProb10P"
# python train.py --experiment_name abcdASTWMostPActionChainProb10P \
#  --model_name_or_path results/abcdASTWMostPActionChainProb10P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_prob_10p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_prob_10p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_noinit_prob.json \
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
# echo "abcdASTWMostPActionChainProb30P"
# python train.py --experiment_name abcdASTWMostPActionChainProb30P \
#  --model_name_or_path results/abcdASTWMostPActionChainProb30P_input_target_t5-small/checkpoint-8686 \
#   --do_predict \
#   --train_file ./data/processed/train_AST_abcd_wmostpaction_chain_prob_30p.json \
#   --validation_file ./data/processed/dev_AST_abcd_wmostpaction_chain_prob_30p.json \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_prob_30p.json \
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
# python eval_abcd_AST_useArgs.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/chainPrior/learned_mdp_8000.dot \
#   --chainPrior False \
#   --experiment_name abcdASTWOAction \
#   --model_name_or_path results/abcdASTWOAction_input_target_t5-small \
#   --do_predict \
#   --test_file ./data/processed/test_AST_abcd-full.json \
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
#   --use_ast_metrics

# python eval_abcd_AST_useArgs.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/chainPrior/learned_mdp_80.dot \
#   --chainPrior True \
#   --experiment_name abcdASTWOAction1P \
#   --model_name_or_path results/abcdASTWOAction1P_input_target_t5-small \
#   --do_predict \
#   --test_file ./data/processed/test_AST_abcd_1p.json \
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
#   --use_ast_metrics


# python eval_abcd_AST_useArgs.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/chainPrior/learned_mdp_800.dot \
#   --chainPrior False \
#   --experiment_name abcdASTWOAction10P \
#   --model_name_or_path results/abcdASTWOAction10P_input_target_t5-small \
#   --do_predict \
#   --test_file ./data/processed/test_AST_abcd_10p.json \
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
#   --use_ast_metrics


python eval_abcd_AST_useArgs.py \
  --automaton_path /research/d5/gds/xywen22/project/llm_framework/chainPrior/learned_mdp_4000.dot \
  --chainPrior False \
  --experiment_name abcdASTWOActionHalf \
  --model_name_or_path results/abcdASTWOActionHalf_input_target_t5-small \
  --do_predict \
  --test_file ./data/processed/test_AST_abcd_half.json \
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
  --use_ast_metrics

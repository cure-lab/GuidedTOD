# set cuda device
# export CUDA_VISIBLE_DEVICES=0,1
# echo "multiwozMultiASTWAction1P"
# python train.py --experiment_name multiwozMultiASTWAction1P \
#  --model_name_or_path results/multiwozMultiASTWAction1P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_1p.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_1p.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_1p.json \
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
# echo "multiwozMultiASTWAction10P"
# python train.py --experiment_name multiwozMultiASTWAction10P \
#  --model_name_or_path results/multiwozMultiASTWAction10P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_10p.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_10p.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_10p.json \
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
#   --num_beams 1


# export CUDA_VISIBLE_DEVICES=6,7
# echo "multiwozMultiASTWAction20P"
# python train.py --experiment_name multiwozMultiASTWAction20P \
#  --model_name_or_path results/multiwozMultiASTWAction20P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_20p.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_20p.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_20p.json \
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
# echo "multiwozMultiASTWAction30P"
# python train.py --experiment_name multiwozMultiASTWAction30P \
#  --model_name_or_path results/multiwozMultiASTWAction30P_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_30p.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_30p.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_30p.json \
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
#   --num_beams 1


# export CUDA_VISIBLE_DEVICES=6,7
# echo "multiwozMultiASTWActionHalf"
# python train.py --experiment_name multiwozMultiASTWActionHalf \
#  --model_name_or_path results/multiwozMultiASTWActionHalf_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_half.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_half.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_half.json \
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
# echo "multiwozMultiASTWActionAll30Epoch"
# python train.py --experiment_name multiwozMultiASTWActionAll30Epoch \
#  --model_name_or_path results/multiwozMultiASTWActionAll30Epoch_input_target_t5-small \
#   --do_predict \
#   --train_file ./data/processed/train_multiAST_multiwoz_waction_all.json \
#   --validation_file ./data/processed/validation_multiAST_multiwoz_waction_all.json \
#   --test_file ./data/processed/test_multiAST_multiwoz_waction_all.json \
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
#   --num_beams 1
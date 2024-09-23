
# export CUDA_VISIBLE_DEVICES=7
# echo "abcdASTWMostPActionChain30P"
# python eval_abcd_AST_directLoading.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
#   --chainPrior False \
#   --model_name_or_path results/abcdASTWMostPActionChain30P_input_target_t5-small \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_30p.json \
#   --alpha 0.05

# echo "abcdASTWAction30P"
# python eval_abcd_AST_directLoading.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
#   --chainPrior False \
#   --model_name_or_path results/abcdASTWAction30P_input_target_t5-small \
#   --test_file ./data/processed/test_AST_abcd_waction_30p.json \
#   --alpha 0.05

# export CUDA_VISIBLE_DEVICES=7
# echo "abcdASTWPAction30P"
# python eval_abcd_AST_directLoading.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
#   --chainPrior True \
#   --model_name_or_path results/abcdASTWPAction30P_input_target_t5-small \
#   --test_file ./data/processed/test_AST_abcd_wpaction_30p.json \
#   --alpha 0.5


# export CUDA_VISIBLE_DEVICES=7
# echo "abcdASTWAction1P"
# python eval_abcd_AST_directLoading.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
#   --chainPrior True \
#   --model_name_or_path results/abcdASTWAction1P_input_target_t5-small \
#   --test_file ./data/processed/test_AST_abcd_waction_1p.json \
#   --alpha 0.1

export CUDA_VISIBLE_DEVICES=7
echo "abcdASTWMostPActionChainAll"
python eval_abcd_AST_directLoading.py \
  --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
  --chainPrior False \
  --model_name_or_path results/abcdASTWMostPActionChainAll_input_target_t5-small \
  --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_all.json \
  --alpha 0.1
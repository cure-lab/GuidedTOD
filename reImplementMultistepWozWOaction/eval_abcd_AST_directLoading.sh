
# echo "abcdASTWMostPActionChain30P"
# python eval_abcd_AST_directLoading.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
#   --chainPrior False \
#   --model_name_or_path results/abcdASTWMostPActionChain30P_input_target_t5-small \
#   --test_file ./data/processed/test_AST_abcd_wmostpaction_chain_30p.json \
#   --alpha 0.05


# echo "abcdMultiASTWAction30P"
# python eval_abcd_AST_directLoading.py \
#   --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
#   --chainPrior False \
#   --model_name_or_path results/abcdMultiASTWAction30P_input_target_t5-small \
#   --test_file ./data/processed/test_multiAST_abcd_waction_30p.json \
#   --alpha 0.05


echo "abcdMultiASTWActionAll"
python eval_abcd_AST_directLoading.py \
  --automaton_path /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/mdp_guildline_half.dot \
  --chainPrior False \
  --model_name_or_path results/abcdMultiASTWActionAll_input_target_t5-small \
  --test_file ./data/processed/test_multiAST_abcd_waction_all.json \
  --alpha 0.05

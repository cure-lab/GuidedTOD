# # w/o action
# python abcd.py \
#     --exp_name abcd \
#     --test_file /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd_woaction_flow_all.json

# with action
python abcd.py \
    --exp_name abcd \
    --waction \
    --test_file /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd_waction_flow_all.json \
    --file_suffix wactionGPT4TurboTrial1
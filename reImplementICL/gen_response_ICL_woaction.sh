# abcd w/o action
# echo "abcd w/o action"
# python gen_response.py \
#     --exp_name abcd \
#     --test_file /research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd_woaction_flow_all.json \
#     --file_suffix woactionGPT4TurboTrial1



# multiwoz w/o action
# gpt_model can be:
# gpt-3.5-turbo
# gpt-4-turbo
# gpt-4

# echo "multiwoz w/o action gpt-3.5-turbo"
# python gen_response.py \
#     --dataset multiwoz \
#     --test_file /research/d5/gds/xywen22/project/llm_framework/AST_multiwoz_part/data/processed/test_AST_multiwoz_woaction_name_all.json \
#     --file_suffix Trial1 \
#     --gpt_model gpt-3.5-turbo \

# echo "multiwoz w/o action gpt-4-turbo"
# python gen_response.py \
#     --dataset multiwoz \
#     --test_file /research/d5/gds/xywen22/project/llm_framework/AST_multiwoz_part/data/processed/test_AST_multiwoz_woaction_name_all.json \
#     --file_suffix Trial1 \
#     --gpt_model gpt-4-turbo \

echo "multiwoz w/o action gpt-4"
python gen_response.py \
    --dataset multiwoz \
    --test_file /research/d5/gds/xywen22/project/llm_framework/AST_multiwoz_part/data/processed/test_AST_multiwoz_woaction_name_all.json \
    --file_suffix Trial1 \
    --gpt_model gpt-4 \
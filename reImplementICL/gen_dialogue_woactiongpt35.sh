# exp_name: abcd or multiwoz
# python gen_dialogue.py \
#     --dataset abcd \
#     --model gpt35 \
#     --test_file responses/abcd_wactionGPT3.5Trial1.json \
#     --file_suffix Trial1

python gen_dialogue.py \
    --dataset multiwoz \
    --model gpt35 \
    --test_file responses/multiwoz_woaction_gpt-3.5-turbo_Trial1.json \
    --file_suffix GPT35TurboTrial2
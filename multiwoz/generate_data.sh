################################
# Creating dataset for Multi-WOZ
################################

# echo "Creating dataset for MultiWOZ considering the actions in the workflow only"
# python generate_dataset_multiwoz_WO_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all

# echo "Creating dataset for MultiWOZ considering the actions in the workflow only and extract most possible actions from the chained prior"
# python generate_dataset_multiwoz_w_mostpaction.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all


# echo "Creating dataset for MultiWOZ considering the actions for each turn"
# python generate_dataset_multiwoz_turn_WO_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all


# echo "Creating dataset for MultiWOZ adding previous actions to the context, for turn base action"
# python generate_dataset_multiwoz_turn_W_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all

# echo "Creating dataset for MultiWOZ adding previous actions to the context"
# python generate_dataset_multiwoz_w_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 4000


echo "Creating dataset for MultiWOZ without adding previous actions to the context, but having flow"
python generate_dataset_multiwoz_WO_action_name_flow.py \
    --raw_data_folder ./data/raw \
    --processed_data_folder ./data/processed \
    --sample_numbers all
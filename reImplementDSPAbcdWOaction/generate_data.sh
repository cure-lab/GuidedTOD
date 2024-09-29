# python generate_dataset_incremental.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed


# sample numbers can be:
# 80, 800, 1600, 2400, 4000, all
# all means using all the data to create the dataset
# python generate_dataset_WO_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 4000

# python generate_dataset_W_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 4000

# echo "Generating dataset with possible actions"
# python generate_dataset_W_paction.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 80

# echo "Generating dataset with possible actions (upper bound)"
# python generate_dataset_W_paction_upperbound.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 2400

# echo "Generating dataset with possible actions and sequence actions (upper bound)"
# python generate_dataset_W_paction_sequence_upperbound.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 2400

# echo "Generating dataset with possible actions extracted from the chain"
# python generate_dataset_W_paction_chain.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers 800

echo "Generating dataset with possible actions extracted from the chain, whose sum of the action frequency is greater than 0.8"
python generate_dataset_W_mostpaction_chain.py \
    --raw_data_folder ./data/raw \
    --processed_data_folder ./data/processed \
    --sample_numbers all

# echo "Generating dataset with possible actions extracted from the chain, whose sum of the action frequency is greater than 0.8, and augmented"
# python generate_dataset_W_mostpaction_chain_aug.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all
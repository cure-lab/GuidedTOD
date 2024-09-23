# python generate_dataset_incremental.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed


# sample numbers can be:
# 80, 800, 1600, 2400, 4000, all
# all means using all the data to create the dataset
# python generate_dataset_WO_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all

# python generate_dataset_W_action.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all

# sample numbers can be:
# 80, 800, 1600, 2400, 4000, all
# python generate_dataset_W_action_multistep.py \
#     --raw_data_folder ./data/raw \
#     --processed_data_folder ./data/processed \
#     --sample_numbers all

python generate_dataset_WO_action_multistep.py \
    --raw_data_folder ./data/raw \
    --processed_data_folder ./data/processed \
    --sample_numbers all
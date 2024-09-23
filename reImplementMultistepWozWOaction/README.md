# LLM Framework for Goal-oriented Dialogue Systems

The code base relies on the huggingface transformer library.

# Data
In this work we use two dataset ABCD (Chen et al., 2021) and MultiWOZ 2.2 (Zang et al., 2020)(pending).

### Create folder Structure
Create a following folder structure to contain all the data
```
<Project Directory>/
└── data/
    ├── raw 
    └── processed 
```

```shell
mkdir -p data/raw
mkdir -p data/processed
```

### Copy Action Mapping files
In this work we use a mapping for the action names to convert them to a human written names (e.g., "pull up customer account" instead of "pull-up-account").
This code base includes the mapping that were use for all the experiments in our work for both datasets.

```shell
cp ${Clone_Directory}/resources/abcd_action_mappings.json data/raw
cp ${Clone_Directory}/resources/multiwoz_action_mappings.json data/raw
```



### Download ABCD Dataset 
Since ABCD is not on huggingface datasets, we need to download it manually:

```shell
cd data/raw
wget https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/guidelines.json
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/ontology.json
wget https://raw.githubusercontent.com/asappresearch/abcd/master/data/utterances.json
gunzip abcd_v1.1.json.gz
```

### Install requirements
```shell
# Enable you virtual env
# Chained prior module relies on aalpy
pip install -r requirements.txt
```

### Create Datasets for ABCD

```shell
bash generate_data.sh 
```

Once the script above runs successfully, you should see the following files in the processed data folder
```
<Project Directory>/
└── data/
    └── processed 
       ├── train_workflow_discovery_abcd.json 
       ├── dev_workflow_discovery_abcd.json 
       ├── test_workflow_discovery_abcd.json 
       ├── train_AST_abcd.json 
       ├── dev_AST_abcd.json 
       ├── test_AST_abcd.json 
       ├── train_CDS_abcd.json 
       ├── dev_CDS_abcd.json 
       ├── test_CDS_abcd.json 
       ├── train_workflow_discovery_multiwoz.json 
       ├── validation_workflow_discovery_multiwoz.json 
       └── test_workflow_discovery_multiwoz.json 
```

# Train Policy Model

### Run

```shell
# set cuda device
export CUDA_VISIBLE_DEVICES=2,3
python train.py --experiment_name abcdASTWOActionFull \
 --model_name_or_path t5-small \
  --do_eval \
  --do_predict \
  --num_train_epochs 100 \
  --train_file ./data/processed/train_AST_abcd-full.json \
  --validation_file ./data/processed/dev_AST_abcd-full.json \
  --test_file ./data/processed/test_AST_abcd-full.json \
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
  --use_ast_metrics \
  --use_fast_tokenizer False
```

### Evaluate

```shell
# set cuda device
export CUDA_VISIBLE_DEVICES=6,7
python train.py --experiment_name abcdASTWAction \
 --model_name_or_path results/abcdASTWAction_input_target_t5-small/checkpoint-30500 \
  --do_predict \
  --train_file ./data/processed/test_AST_abcd_50.json \
  --validation_file ./data/processed/dev_AST_abcd-full.json \
  --test_file ./data/processed/test_AST_abcd_w_action_full.json \
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
  --use_ast_metrics \
  --num_beams 4
```
#### Note:
- The `--num_beams` parameter is used to set the number of beams for beam search, it is required to be set to 4 for evaluation.
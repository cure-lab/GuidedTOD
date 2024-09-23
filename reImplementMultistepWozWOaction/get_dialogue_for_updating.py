'''
Set the openai GPT-4
'''
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')
import json
clientGPT4 = OpenAI(api_key="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")
clientGPT3_5 = OpenAI(api_key="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")

'''
Set the AST module for predict the next action: 
For demo, use the model for SGD dataset 
'''
"""
Reference: https://github.com/huggingface/transformers/tree/main/examples/pytorch

Adapted from huggingface Transformers
"""

import logging
import os
import sys
from pathlib import Path
import time

import datasets
import transformers
import transformers.trainer_utils as hf_trainer_utils
import numpy as np
import nltk  # Here to have a nice missing dependency error message early on

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    set_seed,
    MBartTokenizer,
    MBartTokenizerFast,
)

from src.data.data_args import DataArguments
from src.data.dataset_loader import DatasetLoader
from src.data.utils import group_col_name
from src.metrics import create_compute_metric_fct, verify_nltk
from src.model.hf_model_args import HfModelArguments
from src.hf_training.hf_training_args import HfSeq2SeqTrainingArgs

logger = logging.getLogger(__name__)

def train(trainer, train_dataset, training_args):
    logger.info("*** train ***")

    check_point = get_resume_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=check_point)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def do_eval(trainer, validation_dataset, max_length, num_beams):
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

    metrics["eval_samples"] = len(validation_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def do_predict(trainer, test_dataset, tokenizer, training_args, data_args, model_args, max_length, num_beams):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def decode(preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        model_path = Path(model_args.model_name_or_path)
        file_name = "pred_mwoz.txt" if training_args.is_mwoz else "preds_test_set.txt"
        if not model_path.exists():
            # model name
            preds_file_path = Path(training_args.output_dir) / file_name
        else:
            preds_file_path = model_path / file_name

        with preds_file_path.open("w") as f:
            for pred, label in zip(decoded_preds, decoded_labels):
                label = label.replace("\n", " ")
                pred = pred.replace("\n", " ")
                f.write(f"{pred}\t{label}" + "\n")

        return decoded_preds, decoded_labels
    logger.info("*** Predict ***")

    metrics = {}
    predictions = []
    if group_col_name in test_dataset.column_names:
        group_idx = 0

        while True:
            group_dataset = test_dataset.filter(lambda x: x[group_col_name] == group_idx)
            if group_dataset.num_rows == 0:
                # no groups left
                break
            logger.info("Predicting on test group %d", group_idx)

            predict_results = trainer.predict(
                group_dataset,
                metric_key_prefix=f"predict_group_{group_idx}",
                max_length=max_length,
                num_beams=num_beams
            )
            metrics.update(predict_results.metrics)
            metrics[f"predict_samples_group_{group_idx}_size"] = len(group_dataset)

            group_idx += 1

            predictions.append(predict_results.predictions)

        for key in ["loss", "rouge1", "rouge2", "rougeL"]:
            metrics[f"overall_predict_{key}"] = round(
                sum([metrics[f"predict_group_{idx}_{key}"] for idx in range(group_idx)]) / group_idx, 4
            )
    else:
        '''
        here
        '''
        # print("test_dataset.column_names: ", test_dataset.column_names)
        # print("test_dataset: ", test_dataset)
        # print("test_dataset[:2]: ", test_dataset[:2])
        # sample_test_dataset = test_dataset.filter(lambda x: x["sample_id"] in [0, 1, 3])
        # print("sample_test_dataset[\"sample_id\"]: ", sample_test_dataset["sample_id"])
        # sample_test_dataset["sample_id"] = [0, 1, 2]
        # print("sample_test_dataset[\"sample_id\"]: ", sample_test_dataset["sample_id"])
        # sample_test_dataset["input_ids"] = [sample_test_dataset["input_ids"][0], sample_test_dataset["input_ids"][1], [1,2,3,4,5]]
        # print("sample_test_dataset[\"input_ids\"]: ", sample_test_dataset["input_ids"])

        # print("sample_test_dataset: ", sample_test_dataset)
        # print(test_dataset["sample_id"])
        # print(test_dataset["input_ids"])
        # print(test_dataset["labels"])
        
        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="test", max_length=max_length, num_beams=num_beams
        )
        # print("predict_results: ", predict_results)
        # print("predict_results.predictions: ", predict_results.predictions)
        metrics = predict_results.metrics
        metrics["predict_samples_size"] = len(test_dataset)

    # trainer.log(metrics)
    # trainer.log_metrics("test", metrics)
    # trainer.save_metrics("test", metrics)

    return decode(predict_results.predictions, test_dataset["labels"])


def load_model(model_args, data_args, tokenizer):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Forcing the generation min lenght, to avoid models preset for summarization tasks that are usually high
    config.min_length = 5

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("summarization_cnn", {}))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id["en_XX"]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("en_XX")

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    return model


def get_resume_checkpoint(training_args):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    last_checkpoint = get_last_checkpoint(training_args)
    if last_checkpoint is not None:
        checkpoint = last_checkpoint

    return checkpoint


def get_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = hf_trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming hf_training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def create_data_collector(model, tokenizer, training_args, data_args):
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )


def setup_wandb(training_args):
    if training_args.use_wandb:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project_name
        training_args.run_name = training_args.experiment_name


def get_args():
    parser = HfArgumentParser((HfModelArguments, DataArguments, HfSeq2SeqTrainingArgs))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    name_parts = [training_args.experiment_name]
    name_parts.extend([data_args.text_column, data_args.summary_column])

    name_parts.append(model_args.model_name_or_path)

    training_args.experiment_name = "_".join(name_parts)

    training_args.output_dir = str(Path(training_args.output_dir).joinpath(training_args.experiment_name))

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    return data_args, model_args, training_args

# def hf_run():
data_args, model_args, training_args = get_args()

setup_wandb(training_args)

setup_logging(training_args)

verify_nltk()

logger.warning(
    "Process rank: %s, device: %s, n_gpu: % distributed hf_training: %s 16-bits hf_training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

set_seed(training_args.seed)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

datasets_loader = DatasetLoader(data_args, training_args, tokenizer)
train_dataset, validation_dataset, test_dataset = datasets_loader.load_datasets()

model = load_model(model_args, data_args, tokenizer)

if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning(
        "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
        "`%s`. This will lead to loss being calculated twice and will take up more memory",
        model.__class__.__name__,
    )
metric_fct = create_compute_metric_fct(tokenizer, data_args, training_args, model_args)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=create_data_collector(model, tokenizer, training_args, data_args),
    compute_metrics=metric_fct if training_args.predict_with_generate else None,
)

if training_args.do_train:
    train(trainer, train_dataset, training_args)

max_length = (
    training_args.generation_max_length
    if training_args.generation_max_length is not None
    else data_args.val_max_target_length
)
num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

# if training_args.do_eval:
#     do_eval(trainer, validation_dataset, max_length, num_beams)

# if training_args.do_predict:
#     results_pred, results_label = do_predict(trainer, test_dataset, tokenizer, training_args, data_args, model_args, max_length, num_beams)
    # print("results_pred: ", results_pred)
    # print("results_label: ", results_label)

import re

def postprocess_predictions(prediction_str):
    # print("prediction_str: ", prediction_str)
    match = re.match(r"(.*)\[(.*)]", prediction_str)
    if match:
        # action w/ value
        action_name = match.group(1).strip()
        slot_str = match.group(2)
        slot_str = slot_str.replace(";", ",")
        slots = [s.strip() for s in slot_str.split(",")]
        for i in range(len(slots)):
            if slots[i].endswith(">") and not slots[i].startswith("<"):
                # add "<" to the beginning of the slot
                slots[i] = "<" + slots[i]
            if slots[i].startswith("<") and not slots[i].endswith(">"):
                # add ">" to the end of the slot
                slots[i] = slots[i] + ">"
        post_str = action_name + " " + "[" + ", ".join(slots) + "]"
        # print("post_str: ", post_str)
        return post_str
    else:
        return prediction_str

def parse_ast_prediction(prediction_str):
    match = re.match(r"(.*)\[(.*)]", prediction_str)
    if match:
        # action w/ value
        action_name = match.group(1).strip()
        slot_str = match.group(2)
        slot_str = slot_str.replace(";", ",")
        slots = [s.strip() for s in slot_str.split(",")]
        for i in range(len(slots)):
            if slots[i].endswith(">") and not slots[i].startswith("<"):
                # add "<" to the beginning of the slot
                slots[i] = "<" + slots[i]
            if slots[i].startswith("<") and not slots[i].endswith(">"):
                # add ">" to the end of the slot
                slots[i] = slots[i] + ">"
    else:
        action_name = "MISSING"
        slots = ["MISSING"]
    return action_name, slots

def compute_ast_acc_metrics(predictions, labels, convo_ids, turn_ids):
    # print("predictions: ", predictions)
    # print("labels: ", labels)
    """Adapted from ABCD. """
    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []

    for pred, label in zip(predictions, labels):

        action_label, values_label = parse_ast_prediction(label)
        values_label.sort()
        # for value in values_label:
        #     action_labels.append(action_label)
        #     value_labels.append(value)
        action_labels.append(action_label)
        value_labels.append(values_label)

        action_pred, values_pred = parse_ast_prediction(pred)
        values_pred.sort()

        if len(values_pred) > len(values_label):
            values_pred = [v for v in values_label if v in values_pred]
        if len(values_pred) < len(values_label):
            values_pred.extend(["MISSING"] * (len(values_label) - len(values_pred)))

        # for value in values_pred:
        #     action_preds.append(action_pred)
        #     value_preds.append(value)
        action_preds.append(action_pred)
        value_preds.append(values_pred)
    # since the np.array tends to convert the list of lists into a 2D array (plain list), we need to append a dummy value to the end
    # making sure the processed list will be the list of List objects
    if len(value_labels) == 1 and type(value_labels[0]) == str:
        value_labels = [value_labels]
        value_labels.append(["none"])
    if len(value_preds) == 1 and type(value_preds[0]) == str:
        value_preds = [value_preds]
        value_preds.append(["none"])
    if len(value_labels) == 1 and type(value_labels[0]) == list:
        value_labels.append(["none"])
    if len(value_preds) == 1 and type(value_preds[0]) == list:
        value_preds.append(["none"])
    value_preds[-1].append("none")
    value_labels[-1].append("none")

    action_labels_arrary = np.array(action_labels)
    action_preds_arrary = np.array(action_preds)
    print(f"action_labels_arrary: {action_labels_arrary}")
    print(f"action_preds_arrary: {action_preds_arrary}")
    action_match = action_labels_arrary == action_preds_arrary
    # print(f"action_match: {action_match}")
    # print()
    action_acc = sum(action_match) / float(len(action_labels))

    value_labels_arrary = np.array(value_labels)
    value_preds_arrary = np.array(value_preds)
    print(f"value_labels_arrary: {value_labels_arrary}")
    print(f"value_preds_arrary: {value_preds_arrary}")
    value_match = value_labels_arrary == value_preds_arrary
    # print("value_match: ", value_match)
    value_acc = sum(value_match) / float(len(action_labels))

    joint_match = action_match & value_match
    joint_acc = sum(joint_match) / float(len(action_labels))

    # group by convo_ids
    unique_convo_ids = list(set(convo_ids))
    # print("unique_convo_ids: ", unique_convo_ids)
    conversations = {}
    for uci in unique_convo_ids:
        turns, correctness = [], []
        correctness_action, correctness_value = [], []
        row_id = 0
        for convo_id, turn_count in zip(convo_ids, turn_ids):
            if convo_id == uci:
                turns.append(turn_count)
                correct = False
                correct_action = False
                correct_value = False
                action_right = action_match[row_id]
                value_right = value_match[row_id]
                print(f"action_right: {action_right}, value_right: {value_right}")
                
                if action_right:
                    correct_action = True
                else:
                    correct_action = False
                
                if value_right:
                    correct_value = True
                else:
                    correct_value = False

                if action_right and value_right:
                    correct = True
                else:
                    correct = False

                correctness.append(correct)
                correctness_action.append(correct_action)
                correctness_value.append(correct_value)
            row_id += 1

        # sort by turn_counts
        ordered = [cor for _, cor in sorted(zip(turns, correctness), key=lambda tc: tc[0])]
        ordered_action = [cor for _, cor in sorted(zip(turns, correctness_action), key=lambda tc: tc[0])]
        ordered_value = [cor for _, cor in sorted(zip(turns, correctness_value), key=lambda tc: tc[0])]
        conversations[uci] = [ordered, ordered_action, ordered_value]

    # print("ordered: ", ordered)
    # print("ordered_action: ", ordered_action)
    # print("ordered_value: ", ordered_value)

    # count how many correct
    turn_score, turn_correct = 0, 0
    turn_score_action, turn_correct_action = 0, 0
    turn_score_value, turn_correct_value = 0, 0
    em_joint, em_action, em_value = [], [], []
    my_scores = []
    for convo_id, itm in conversations.items():
        convo_correctness = itm[0]
        convo_correctness_action = itm[1]
        convo_correctness_value = itm[2]

        # calculate EM
        if sum(convo_correctness) == len(convo_correctness):
            em_joint.append(True)
        else:
            em_joint.append(False)
        if sum(convo_correctness_action) == len(convo_correctness_action):
            em_action.append(True)
        else:
            em_action.append(False)
        if sum(convo_correctness_value) == len(convo_correctness_value):
            em_value.append(True)
        else:
            em_value.append(False)
        
        # print(f"convo_id: {convo_id}, convo_correctness: {convo_correctness}")
        current_score = 0
        convo_length = len(convo_correctness)
        # we use turn_id rather than the true turn_count since turn counts will skip numbers
        # when looping through the conversation due to skipping over customer utterances
        for turn_id in range(convo_length):
            num_remaining = convo_length - turn_id

            num_correct = 0
            num_correct_action = 0
            num_correct_value = 0
            # count up how many were predicted correctly
            tmp_turn_id = turn_id
            while tmp_turn_id < convo_length and convo_correctness[tmp_turn_id]:
                num_correct += 1
                tmp_turn_id += 1
            
            tmp_turn_id = turn_id
            while tmp_turn_id < convo_length and convo_correctness_action[tmp_turn_id]:
                num_correct_action += 1
                tmp_turn_id += 1

            tmp_turn_id = turn_id
            while tmp_turn_id < convo_length and convo_correctness_value[tmp_turn_id]:
                num_correct_value += 1
                tmp_turn_id += 1

            if num_correct > 0:
                turn_correct += 1
            if num_correct_action > 0:
                turn_correct_action += 1
            if num_correct_value > 0:
                turn_correct_value += 1
            # normalize by the number of turns remaining
            turn_score += num_correct / num_remaining
            turn_score_action += num_correct_action / num_remaining
            turn_score_value += num_correct_value / num_remaining
            # current_score += num_correct / num_remaining

        # my_scores.append(current_score / convo_length)

    # normalize by total number of turns possible
    '''
    len(convo_ids): 200, len(turn_ids): 200
    '''
    # print(f"len(convo_ids): {len(convo_ids)}, len(turn_ids): {len(turn_ids)}")
    turn_acc = turn_correct / float(len(convo_ids))
    turn_acc_action = turn_correct_action / float(len(convo_ids))
    turn_acc_value = turn_correct_value / float(len(convo_ids))
    final_score = turn_score / float(len(convo_ids))
    final_score_action = turn_score_action / float(len(convo_ids))
    final_score_value = turn_score_value / float(len(convo_ids))
    
    em_action_score = sum(em_action) / float(len(em_action))
    em_value_score = sum(em_value) / float(len(em_value))
    em_joint_score = sum(em_joint) / float(len(em_joint))

    return {
        "EM_action": round(em_action_score, 4),
        "EM_value": round(em_value_score, 4),
        "EM_joint": round(em_joint_score, 4),
        "turn_acc_joint": round(turn_acc, 4),
        "turn_acc_action": round(turn_acc_action, 4),
        "turn_acc_value": round(turn_acc_value, 4),
        "CE_joint": round(final_score, 4),
        "CE_action": round(final_score_action, 4),
        "CE_value": round(final_score_value, 4)
    }

data = []
with open('/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/train_AST_abcd.json', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        data.append(json_data)

context_list = []
distinct_dialogue = {}
distinct_dialogue["dialogue"] = []
distinct_dialogue["pred_action_value"] = []
distinct_dialogue["action_value_label"] = []
distinct_dialogue["convo_ids"] = []
distinct_dialogue["turn_ids"] = []
current_conv_id = 0
counter_success_dialogues = 0

if os.path.exists("data/processed/incremental_data.json"):
    # remove the file
    os.remove("data/processed/incremental_data.json")

for dialogue_i in range(len(data)):
    user_input = data[dialogue_i]['input']
    context = user_input
    # save the context to a tmp json file:
    # {"sample_id": 0, "target": "request time [none]", "input": "Context: hi, could you get me a restaurant booking on the 8th please? ", "target_data": "[\"request time\", [\"none\"]]"}
    save_context = {"sample_id": data[dialogue_i]['sample_id'], "convo_id": data[dialogue_i]['convo_id'], "turn_id": data[dialogue_i]['turn_id'], "target": data[dialogue_i]['target'], "input": context, "target_data": data[dialogue_i]['target_data']}
    if os.path.exists("tmp.json"):
        os.remove("tmp.json")
    # print(tmp_sample)
    with open(f"tmp.json", "a") as w:
        json.dump(save_context, w)
        w.write("\n")

    train_dataset, validation_dataset, test_dataset = datasets_loader.load_datasets()
    result_pred, result_label = do_predict(trainer, test_dataset, tokenizer, training_args, data_args, model_args, max_length, num_beams)

    # Call the LLM model to generate the response
    action = result_pred[-1]

    action = postprocess_predictions(action)

    # print("context: ", context)
    # print("agent: ", action)
    # print("gold: ", data[dialogue_i]['target'])
    # print("-" * 30)
    # print()

    if data[dialogue_i]['convo_id'] != current_conv_id:
        if current_conv_id == 0:
            pass
        else:
            # calculate the CE metric
            print(distinct_dialogue["pred_action_value"])
            print(distinct_dialogue["action_value_label"])
            print(distinct_dialogue["convo_ids"])
            print(distinct_dialogue["turn_ids"])
            metrics = compute_ast_acc_metrics(distinct_dialogue["pred_action_value"], distinct_dialogue["action_value_label"], distinct_dialogue["convo_ids"], distinct_dialogue["turn_ids"])
            
            # print("CE_joint: ", metrics["CE_joint"])
            # print("CE_action: ", metrics["CE_action"])
            # print("CE_value: ", metrics["CE_value"])
            # print(distinct_dialogue["pred_action_value"])
            # print(distinct_dialogue["action_value_label"])
            if metrics["CE_joint"] > 0.5 and metrics["CE_action"] > 0.5 and metrics["CE_value"] > 0.5:
                print("CE_joint: ", metrics["CE_joint"])
                print("CE_action: ", metrics["CE_action"])
                print("CE_value: ", metrics["CE_value"])
                print("EM action: ", metrics["EM_action"])
                print("EM value: ", metrics["EM_value"])
                print("EM joint: ", metrics["EM_joint"])
                print("pred_action_value: ", distinct_dialogue["pred_action_value"])
                print("action_value_label: ", distinct_dialogue["action_value_label"])

                counter_success_dialogues += 1

                for i in range(len(distinct_dialogue["dialogue"])):
                    # print(distinct_dialogue["dialogue"][i]["input"])
                    # print(distinct_dialogue["dialogue"][i]["predicted_action"])
                    # print(distinct_dialogue["dialogue"][i]["target"])
                    # print("-" * 30)
                    with open("data/processed/incremental_data.json", "a") as w:
                        json.dump(distinct_dialogue["dialogue"][i], w)
                        w.write("\n")

                if counter_success_dialogues == 500:
                    break

        distinct_dialogue["dialogue"] = []
        distinct_dialogue["pred_action_value"] = []
        distinct_dialogue["action_value_label"] = []
        distinct_dialogue["convo_ids"] = []
        distinct_dialogue["turn_ids"] = []

        save_context['predicted_action'] = action
        distinct_dialogue["dialogue"].append(save_context)
        distinct_dialogue["pred_action_value"].append(action)
        distinct_dialogue["action_value_label"].append(data[dialogue_i]['target'])
        distinct_dialogue["convo_ids"].append(data[dialogue_i]['convo_id'])
        distinct_dialogue["turn_ids"].append(data[dialogue_i]['turn_id'])
        current_conv_id = data[dialogue_i]['convo_id']
    else:
        save_context['predicted_action'] = action
        distinct_dialogue["dialogue"].append(save_context)
        distinct_dialogue["pred_action_value"].append(action)
        distinct_dialogue["action_value_label"].append(data[dialogue_i]['target'])
        distinct_dialogue["convo_ids"].append(data[dialogue_i]['convo_id'])
        distinct_dialogue["turn_ids"].append(data[dialogue_i]['turn_id'])

print("counter_success_dialogues: ", counter_success_dialogues)
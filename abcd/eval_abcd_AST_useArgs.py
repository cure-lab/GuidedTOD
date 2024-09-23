import re
import json

import transformers
import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoModel,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    set_seed,
    MBartTokenizer,
    MBartTokenizerFast,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor
)
import numpy as np
import torch
from aalpy.utils import load_automaton_from_file, save_automaton_to_file, visualize_automaton, generate_random_dfa
from aalpy.automata import Dfa
from aalpy.SULs import AutomatonSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar, run_KV
import argparse
from tqdm import tqdm
from pathlib import Path
import logging

from src.data.data_args import DataArguments
from src.model.hf_model_args import HfModelArguments
from src.hf_training.hf_training_args import HfSeq2SeqTrainingArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# # load args
# parser = argparse.ArgumentParser()
# parser.add_argument("--automaton_path", type=str, default="t5-small")
# parser.add_argument("--chainPrior", default=False)

# args = parser.parse_args()
# # print all args
# print("args: ", args)

# data_args:  DataArguments(text_column='input', summary_column='target', dataset_name=None, dataset_config_name=None, train_file='./data/processed/train_AST_abcd_10p.json', validation_file='./data/processed/dev_AST_abcd_10p.json', test_file='./data/processed/test_AST_abcd_10p.json', overwrite_cache=False, preprocessing_num_workers=None, max_source_length=1024, max_target_length=256, val_max_target_length=256, pad_to_max_length=False, max_train_samples=None, max_eval_samples=None, max_predict_samples=None, num_beams=None, ignore_pad_token_for_loss=True, source_prefix='Predict AST: ', data_cache_dir=None, original_dialog_column='original_dialog')
# model_args:  HfModelArguments(model_name_or_path='results/abcdASTWOAction10P_input_target_t5-small', config_name=None, tokenizer_name=None, cache_dir=None, use_fast_tokenizer=False, model_revision='main', use_auth_token=False, resize_position_embeddings=None)

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

def get_args():
    parser = HfArgumentParser((HfModelArguments, DataArguments))

    model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # name_parts = [training_args.experiment_name]
    # name_parts.extend([data_args.text_column, data_args.summary_column])

    # name_parts.append(model_args.model_name_or_path)

    # training_args.experiment_name = "_".join(name_parts)

    # training_args.output_dir = str(Path(training_args.output_dir).joinpath(training_args.experiment_name))

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
    return data_args, model_args

data_args, model_args = get_args()

tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

model = load_model(model_args, data_args, tokenizer)
model.to(device)

# load an automaton
automaton = load_automaton_from_file(model_args.automaton_path, automaton_type='mdp')
# print(automaton)
# visualize the automaton
# visualize_automaton(automaton)
automaton = str(automaton)
# print(automaton)

automaton_splits = automaton.split('\n')
# print(automaton_splits)
automaton_states = automaton_splits[1:33]
# ['s0 [label="init"];', 's1 [label="pull-up-account"];']
automaton_transitions = automaton_splits[33:-4]
# ['s0 -> s0  [label="init:1.0"];', 's0 -> s1  [label="action:0.03"];']

state_mapping = {}
for state in automaton_states:
    state_name = state.split(' ')[0]
    state_label = state.split('[label="')[1].split('"];')[0]
    state_mapping[state_name] = state_label

transition_mapping = {}
for transition in automaton_transitions:
    transition_split = transition.split('->')
    source_state = transition_split[0].strip()
    target_state = transition_split[1].strip().split(' ')[0]
    transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
    transition_action = transition_label.split(':')[0]
    transition_freq = float(transition_label.split(':')[1])
    transition_mapping[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)

# from s0 to s31, if some pair of states are not in the transition_mapping, then the frequency is 0
for i in range(32):
    for j in range(32):
        if (state_mapping[f's{i}'], state_mapping[f's{j}']) not in transition_mapping:
            transition_mapping[(state_mapping[f's{i}'], state_mapping[f's{j}'])] = ('unknown', 0.0)

# print("state_mapping: ", state_mapping)
# print("transition_mapping: ", transition_mapping)

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

    action_labels_arrary = np.array(action_labels, dtype=object)
    action_preds_arrary = np.array(action_preds, dtype=object)
    action_match = action_labels_arrary == action_preds_arrary
    action_acc = sum(action_match) / float(len(action_labels))

    value_labels_arrary = np.array(value_labels, dtype=object)
    value_preds_arrary = np.array(value_preds, dtype=object)
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
with open(data_args.test_file, 'r') as file:
    for line in file:
        json_data = json.loads(line)
        data.append(json_data)

# split the data into convo_ids
convos = {}
for i in range(len(data)):
    convo_id = data[i]['convo_id']
    if convo_id not in convos:
        convos[convo_id] = []
    convos[convo_id].append(data[i])

context_list = []
distinct_dialogue = {}
distinct_dialogue["dialogue"] = []
distinct_dialogue["pred_action_value"] = []
distinct_dialogue["action_value_label"] = []
distinct_dialogue["convo_ids"] = []
distinct_dialogue["turn_ids"] = []
current_conv_id = 0
counter_success_dialogues = 0

# if os.path.exists("data/processed/incremental_data.json"):
#     # remove the file
#     os.remove("data/processed/incremental_data.json")

counter = 0
CE_joint = []
CE_action = []
CE_value = []
EM_joint = []
EM_action = []
EM_value = []
for convo_id, convo_data in tqdm(convos.items()):
    # if counter == 100:
    #     break
    # print("convo_id: ", convo_id)
    previous_actions = ['init']
    current_dialogue = {}
    current_dialogue["pred_action_value"] = []
    current_dialogue["action_value_label"] = []
    current_dialogue["convo_ids"] = []
    current_dialogue["turn_ids"] = []
    for i in range(len(convo_data)):
        # print("input: ", convo_data[i]['input'])
        # print("target: ", convo_data[i]['target'])

        input_context = "Predict AST: " + convo_data[i]['input']
        encoder_input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        encoder_input_ids = encoder_input_ids.to(device)

        # lets run beam search using 3 beams
        num_beams = 4
        # define decoder start token ids
        input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        model_kwargs = {
            "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        }

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            device=model.device,
            num_beam_hyps_to_keep=num_beams,
        )

        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ]
        )

        outputs = model.beam_search(
            input_ids, 
            beam_scorer, 
            logits_processor=logits_processor, 
            output_scores=True, 
            return_dict_in_generate = True,
            **model_kwargs
        )

        scores = outputs.sequences_scores.cpu().numpy()
        for j in range(len(scores)):
            scores[j] = np.exp(scores[j])
        # print("scores: ", scores)
        # normalize the scores
        scores = scores / np.sum(scores)

        action_value1 = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        action_value2 = tokenizer.decode(outputs.sequences[1], skip_special_tokens=True)
        action_value3 = tokenizer.decode(outputs.sequences[2], skip_special_tokens=True)
        action_value4 = tokenizer.decode(outputs.sequences[3], skip_special_tokens=True)
        
        action_value1 = postprocess_predictions(action_value1)
        action_value2 = postprocess_predictions(action_value2)
        action_value3 = postprocess_predictions(action_value3)
        action_value4 = postprocess_predictions(action_value4)

        action1 = action_value1.split(' ')[0].strip()
        action2 = action_value2.split(' ')[0].strip()
        action3 = action_value3.split(' ')[0].strip()
        action4 = action_value4.split(' ')[0].strip()

        value1 = action_value1.split(action1)[1].strip()
        value2 = action_value2.split(action2)[1].strip()
        value3 = action_value3.split(action3)[1].strip()
        value4 = action_value4.split(action4)[1].strip()

        try:
            rate1 = transition_mapping[(previous_actions[-1], action1)][1]
        except:
            rate1 = 0.0
        try:
            rate2 = transition_mapping[(previous_actions[-1], action2)][1]
        except:
            rate2 = 0.0
        try:
            rate3 = transition_mapping[(previous_actions[-1], action3)][1]
        except:
            rate3 = 0.0
        try:
            rate4 = transition_mapping[(previous_actions[-1], action4)][1]
        except:
            rate4 = 0.0

        rates = [rate1, rate2, rate3, rate4]
        rates = np.array(rates)
        # print("args.chain_prior: ", args.chain_prior)
        if model_args.chainPrior:
            merge_scores = scores + rates
        else:
            merge_scores = scores
        # print("merge_scores: ", merge_scores)
        # print("scores: ", scores)
        # print("rates: ", rates)
        # find the index of the max value
        max_index = np.argmax(merge_scores)


        # print(f"actions: {action1}, {action2}, {action3}, {action4}")
        # print(f"values: {value1}, {value2}, {value3}, {value4}")
        # print(f"rates: {rate1}, {rate2}, {rate3}, {rate4}")
        # print(f"scores: {scores}")
        # print(f"merge_scores: {merge_scores}")
        # print(f"max_index: {max_index}")
        # print("-" * 30)
        # print()

        selected_action_value = [action_value1, action_value2, action_value3, action_value4][max_index]

        current_dialogue["pred_action_value"].append(selected_action_value)
        current_dialogue["action_value_label"].append(convo_data[i]['target'])
        current_dialogue["convo_ids"].append(convo_data[i]['convo_id'])
        current_dialogue["turn_ids"].append(convo_data[i]['turn_id'])

        previous_actions.append(convo_data[i]['target'].split(' ')[0].strip())
    
    metrics = compute_ast_acc_metrics(current_dialogue["pred_action_value"], current_dialogue["action_value_label"], current_dialogue["convo_ids"], current_dialogue["turn_ids"])

    # print("CE_joint: ", metrics["CE_joint"])
    # print("CE_action: ", metrics["CE_action"])
    # print("CE_value: ", metrics["CE_value"])
    # print("EM action: ", metrics["EM_action"])
    # print("EM value: ", metrics["EM_value"])
    # print("EM joint: ", metrics["EM_joint"])

    CE_joint.append(metrics["CE_joint"])
    CE_action.append(metrics["CE_action"])
    CE_value.append(metrics["CE_value"])
    EM_joint.append(metrics["EM_joint"])
    EM_action.append(metrics["EM_action"])
    EM_value.append(metrics["EM_value"])

    counter += 1

# average the metrics
print()
print("average metrics: ")
print("average CE_joint: ", sum(CE_joint) / len(CE_joint))
print("average CE_action: ", sum(CE_action) / len(CE_action))
print("average CE_value: ", sum(CE_value) / len(CE_value))
print("average EM_joint: ", sum(EM_joint) / len(EM_joint))
print("average EM_action: ", sum(EM_action) / len(EM_action))
print("average EM_value: ", sum(EM_value) / len(EM_value))
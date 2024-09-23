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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load args
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="t5-small")
parser.add_argument("--automaton_path", type=str, default="t5-small")
parser.add_argument("--test_file_path", type=str, default="data/processed/test_AST_abcd-full.json")
parser.add_argument("--chainPrior", required=True, default=False)
parser.add_argument("--alpha", required=True, default=0.5)

args = parser.parse_args()

# print all args
print("args: ", args)

# load an automaton
chained_proir = load_automaton_from_file(args.automaton_path, automaton_type='mc')

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    use_fast=False,
    # revision="main",
    # use_auth_token=None,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    args.model_name_or_path,
)
model.to(device)


data = []
with open(args.test_file_path, 'r') as file:
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


# def normalize_log_probs(log_probs):
#     """
#     Normalizes an array of log probabilities using the Log-Sum-Exp trick
#     to prevent numerical underflow.

#     Parameters:
#     - log_probs: A numpy array containing log probabilities.

#     Returns:
#     - An array of normalized log probabilities.
#     """

#     # Find the maximum log probability to avoid underflow issues
#     max_log_prob = np.max(log_probs)
    
#     # Subtract the max log probability and exponentiate the result
#     probs_stable = np.exp(log_probs - max_log_prob) + 1e-9
    
#     # Sum the stabilized probabilities
#     prob_total_stable = np.sum(probs_stable) + 1e-9
    
#     # Normalize the stabilized probabilities
#     normalized_probs_stable = probs_stable / prob_total_stable
    
#     # Convert back to log probabilities adding the subtracted max_log_prob
#     normalized_log_probs_stable = np.log(normalized_probs_stable) + max_log_prob
    
#     return normalized_log_probs_stable

def normalize_log_probs(log_probs):
    max_log_prob = np.max(log_probs)
    log_probs = log_probs - max_log_prob
    probs = np.exp(log_probs)
    return probs / np.sum(probs)

for convo_id, convo_data in tqdm(convos.items()):
    # if counter == 5:
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
        encoder_input_ids = tokenizer(input_context, return_tensors="pt")
        encoder_input_ids = encoder_input_ids.to(device)

        # lets run beam search using 3 beams
        num_beams = 10
        # define decoder start token ids
        outputs = model.generate(**encoder_input_ids, 
                                 max_new_tokens=256, 
                                 return_dict_in_generate=True, 
                                 output_scores=True, 
                                 do_sample=False, 
                                 num_beams=num_beams, 
                                 num_return_sequences=num_beams
                                )
        
        transition_scores = outputs.sequences_scores.cpu().numpy()
        # print("transition_scores: ", transition_scores)
        # output_length = np.sum(transition_scores < 0, axis=1)
        # output_length = len(transition_scores)
        # length_penalty = model.generation_config.length_penalty
        # reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        reconstructed_scores = transition_scores

        # try:
        #     transition_scores = model.compute_transition_scores(
        #         outputs.sequences, outputs.scores, normalize_logits=False
        #     )
        #     output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        #     length_penalty = model.generation_config.length_penalty
        #     reconstructed_scores = transition_scores.sum(axis=1).cpu().numpy() / (output_length**length_penalty)
        # except:
        #     reconstructed_scores = np.zeros(num_beams)
        #     print(f"cannot compute {i}, set prob to 0")
        
        log_prob = reconstructed_scores

        # print(f'scores: {log_prob}')

        policy_model_action = []
        policy_model_slot = []
        policy_model_prob = []
        policy_max_idx = np.argmax(log_prob)

        # print('+' * 20, 'policy model', '+' * 20)
        for k in range(num_beams):
            pred_str = tokenizer.decode(outputs.sequences[k], skip_special_tokens=True)
            # print(f"pred_str: {pred_str}")
            pred_str = pred_str.split(';')[0].strip()
            # print(f"pred_str: {pred_str}")
            action_name, slots = parse_ast_prediction(pred_str)
            # print(f"policy model top parsed: {action_name}, {slots}; score: {log_prob[k]}")

            policy_model_action.append(action_name)
            policy_model_slot.append(slots)
            if action_name != 'MISSING':
                policy_model_prob.append(log_prob[k])
            else:
                policy_model_prob.append(-10000)

            # if k == 0:
            #     print("policy model top pred str: ", pred_str)
            #     print(f"policy model top parsed: {action_name}, {slots}; score: {log_prob[k]}")

            # if k == policy_max_idx:
            #     print("policy model top pred str: ", pred_str)
            #     print(f"policy model top parsed: {action_name}, {slots}; score: {log_prob[k]}")



        # print("policy_model_prob: ", policy_model_prob)
        # policy_model_prob = normalize_log_probs(policy_model_prob)
        policy_model_prob = list(policy_model_prob)
        # print("policy_model_prob: ", policy_model_prob)
        # print("policy_model_action: ", policy_model_action)

        raw_policy_model_action = policy_model_action.copy()
        raw_policy_model_slot = policy_model_slot.copy()
        raw_policy_model_prob = policy_model_prob.copy()

        # print('+' * 20, 'chained prior', '+' * 20)
        # print(f"context parsed: {previous_actions}")
        init_state = chained_proir.states[0]
        curr_state = init_state

        prev_actions_with_prob = [(curr_state.output, 1)]
        prev_prob = 0
        idx = 0
        while curr_state.output != previous_actions[-1] and idx <= len(previous_actions) - 1:
            idx += 1
            matched = False
            for prosible_state, prob in curr_state.transitions:
                # print(f"prosible_state: {prosible_state.output}, prob: {prob}")
                if prosible_state.output == previous_actions[idx]:
                    # print(f"prev_actions[idx]: {prev_actions[idx]}, curr_state.output: {curr_state.output}")
                    prev_actions_with_prob.append((prosible_state.output, np.log(prob)))
                    prev_prob += np.log(prob)
                    curr_state = prosible_state
                    matched = True
            
            if matched == False:
                break
        # print(f"context parsed prob: {prev_actions_with_prob}, total previous prob: {prev_prob}")

        chained_prior_pred = []
        chained_prior_prob = []
        for state in chained_proir.states:
            # print(f"state: {state.output}")
            if state.output == previous_actions[-1]:
                for prosible_state, prob in state.transitions:
                    chained_prior_pred.append(prosible_state.output)
                    chained_prior_prob.append(np.log(prob) + prev_prob)

        # chained_prior_prob = normalize_log_probs(chained_prior_prob)
        chained_prior_prob = list(chained_prior_prob)
        # print("chained_prior_prob: ", chained_prior_prob)
        # print()
        # print(f"possible next: {chained_prior_pred}, prob: {chained_prior_prob}")

        alpha = float(args.alpha)
        for k, action in enumerate(policy_model_action):
            # print(f"policy model action: {action}, prob: {policy_model_prob[i]}")
            # if action == 'MISSING':
            #     defualt_next = np.argmax(np.array(chained_prior_prob))
            #     policy_model_action[k] = chained_prior_pred[defualt_next]
            #     policy_model_prob[k] = chained_prior_prob[defualt_next] * 1000
            # else:
            for j, c_action in enumerate(chained_prior_pred):
                if action == c_action:
                    # print(f"policy model prob: {policy_model_prob[k]}, chained prior prob: {chained_prior_prob[j]}")
                    # policy_model_prob[k] = policy_model_prob[k] + chained_prior_prob[j] * 100
                    policy_model_prob[k] = policy_model_prob[k] * (1 - alpha) + chained_prior_prob[j] * alpha
        
        # for k, action in enumerate(chained_prior_pred):
        #     if action not in policy_model_action:
        #         policy_model_action.append(action)
        #         policy_model_slot.append(['MISSING'])
        #         policy_model_prob.append(chained_prior_prob[k] * 1000)

        # print("args.chain_prior: ", args.chain_prior)
        if 'true' in args.chainPrior.lower():
            # print('+' * 20, 'chained prior + policy model', '+' * 20)
            # for i in range(len(policy_model_action)):
            #     print(f"MERGED top parsed: {policy_model_action[i]}, {policy_model_slot[i]}; score: {policy_model_prob[i]}")
            top_idx = np.argmax(np.array(policy_model_prob))
            # print("MERGED top pred str: ", f"{policy_model_action[top_idx]} {policy_model_slot[top_idx]}")
            # print(f"MERGED top parsed: {policy_model_action[top_idx]}, {policy_model_slot[top_idx]}; score: {policy_model_prob[top_idx]}")
            # print()
        elif 'false' in args.chainPrior.lower():
            # print('+' * 20, 'policy model', '+' * 20)
            top_idx = np.argmax(np.array(raw_policy_model_prob))
            # print("policy model top pred str: ", f"{raw_policy_model_action[top_idx]} {raw_policy_model_slot[top_idx]}")
            # print(f"policy model top parsed: {raw_policy_model_action[top_idx]}, {raw_policy_model_slot[top_idx]}; score: {raw_policy_model_prob[top_idx]}")
        else:
            raise ValueError("chainPrior must be either true or false")

        selected_action = policy_model_action[top_idx]
        selected_slot = policy_model_slot[top_idx]
        selected_action_value = selected_action + " [" + ", ".join(selected_slot) + "]"
        # print(f"selected_action_value: {selected_action_value}")

        # print(f"selected_action_value: {selected_action_value}")
        # print(f"target: {convo_data[i]['target']}")
        label_action_value = convo_data[i]['target'].split(';')[0].strip()
        # print(f"label_action_value: {label_action_value}")
        current_dialogue["pred_action_value"].append(selected_action_value)
        current_dialogue["action_value_label"].append(label_action_value)
        current_dialogue["convo_ids"].append(convo_data[i]['convo_id'])
        current_dialogue["turn_ids"].append(convo_data[i]['turn_id'])

        previous_actions.append(label_action_value.split(' ')[0].strip())
    
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
print("average CE_action: ", sum(CE_action) / len(CE_action))
print("average CE_value: ", sum(CE_value) / len(CE_value))
print("average CE_joint: ", sum(CE_joint) / len(CE_joint))
print("average EM_action: ", sum(EM_action) / len(EM_action))
print("average EM_value: ", sum(EM_value) / len(EM_value))
print("average EM_joint: ", sum(EM_joint) / len(EM_joint))
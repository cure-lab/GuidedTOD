import re
import sys
from filelock import FileLock
from pathlib import Path
from copy import deepcopy

import jsonlines
from tqdm import tqdm
import numpy as np
import nltk  # Here to have a nice missing dependency error message early on
from nltk.corpus import stopwords
from transformers.file_utils import is_offline_mode
from bert_score import score


def verify_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("stopwords")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def remove_stop_words(string):
    filtered_words = [word for word in string.split(" ") if word not in stopwords.words("english")]
    return " ".join(filtered_words)


def is_action_name_match(target_action_name, predicted_action_name, use_bert_score=False):
    if use_bert_score:
        _, _, F1 = score([target_action_name], [predicted_action_name], lang="en")
        return F1 >= 95.0
    else:
        # We assume stop word does not change the action for our case (e.g., offer promo code is the same as offer a promo code)
        target_action_name = remove_stop_words(target_action_name)
        predicted_action_name = remove_stop_words(predicted_action_name)

        match = target_action_name == predicted_action_name

    return match


def is_slot_values_match(target_slots, predicted_slots):
    # we assume that predicting "the museum" is the same as "museum"
    target_slots = [remove_stop_words(s) for s in target_slots]
    predicted_slots = [remove_stop_words(s) for s in predicted_slots]

    not_found_count = len(target_slots)
    for target_slot in target_slots:
        if isinstance(target_slot, str):
            target_slot = [target_slot]
        for t in target_slot:
            if t in predicted_slots:
                not_found_count -= 1
                break

    match = not_found_count == 0
    return match


def is_flow_action_match(target, prediction, action_only=False, use_bert_score=False):
    target_action_name, target_action_slots = target
    predicted_action_name, predicted_action_slots = prediction

    if not is_action_name_match(target_action_name, predicted_action_name, use_bert_score=use_bert_score):
        return False

    if not action_only and not is_slot_values_match(target_action_slots, predicted_action_slots):
        return False

    return True


def is_exact_match_flow(target_flow, predicted_flow, action_only=False, use_bert_score=False):
    if len(target_flow) != len(predicted_flow):
        # If the length is not the same no need to go further, this will be covered by the CE metric.
        return False

    for target_action, predicted_action in zip(target_flow, predicted_flow):
        if not is_flow_action_match(
            target_action, predicted_action, action_only=action_only, use_bert_score=use_bert_score
        ):
            return False

    return True


def compute_flow_cascading_evaluation(targets, predictions, action_only=False, use_bert_score=False):
    targets = deepcopy(targets)
    predictions = deepcopy(predictions)

    scores = []
    for prediction, target in tqdm(zip(predictions, targets), total=len(targets)):
        if len(prediction) > len(target):
            prediction = [v for v in target if v in prediction]
        if len(prediction) < len(target):
            prediction.extend([["Missing", []]] * (len(target) - len(prediction)))

        current_score = 0
        length = len(target)
        for turn_id in range(length):
            num_remaining = length - turn_id

            num_correct = 0
            # count up how many were predicted correctly
            while turn_id < length and is_flow_action_match(
                target[turn_id], prediction[turn_id], action_only=action_only, use_bert_score=use_bert_score
            ):
                num_correct += 1
                turn_id += 1

            current_score += num_correct / num_remaining

        scores.append(current_score / length)

    return sum(scores) / len(scores)


def compute_flow_cascading_evaluation_w_aga(targets, predictions, use_bert_score=False):
    targets = deepcopy(targets)
    predictions = deepcopy(predictions)

    scores = []
    for prediction, target in tqdm(zip(predictions, targets), total=len(targets)):
        if len(prediction) > len(target):
            prediction = [v for v in target if v in prediction]
        if len(prediction) < len(target):
            prediction.extend([["Missing", []]] * (len(target) - len(prediction)))

        current_score = 0
        length = len(target)
        for turn_id in range(length):
            num_remaining = length - turn_id

            num_correct = 0
            # count up how many were predicted correctly
            while turn_id < length and is_flow_action_match(
                target[turn_id], prediction[turn_id], use_bert_score=use_bert_score
            ):
                num_correct += 1
                turn_id += 1

            current_score += num_correct / num_remaining

        scores.append(current_score / length)

    return sum(scores) / len(scores)


def compute_exact_match(targets, predictions, action_only=False, use_bert_score=False):
    targets = deepcopy(targets)
    predictions = deepcopy(predictions)

    exact_match_count = 0
    for target, prediction in tqdm(zip(targets, predictions), total=(len(targets))):
        if is_exact_match_flow(target, prediction, action_only=action_only, use_bert_score=use_bert_score):
            exact_match_count += 1
    return exact_match_count / float(len(targets))


def compute_metrics(targets, predictions, use_bert_score=False):
    metrics = {
        "EM": compute_exact_match(targets, predictions, use_bert_score=use_bert_score),
        "CE": compute_flow_cascading_evaluation(targets, predictions, use_bert_score=use_bert_score),
        "EM_action_only": compute_exact_match(targets, predictions, action_only=True, use_bert_score=use_bert_score),
        "CE_action_only": compute_flow_cascading_evaluation(
            targets, predictions, action_only=True, use_bert_score=use_bert_score
        ),
    }

    return metrics


def parse_cds_prediction(prediction_str):
    parts = prediction_str.split(";")
    intent = parts[0]
    next_step = "MISSING"
    action = ["MISSING", ["MISSING"]]
    next_utterance = "MISSING"
    if len(parts) > 1:
        next_step = parts[1].strip()
    if len(parts) > 2:
        match = re.match(r"(.*)\[(.*)]", parts[2])
        if match:
            # action w/ value
            action_name = match.group(1).strip()
            slot_str = match.group(2)
            slot_str = slot_str.replace(";", ",")
            slots = [s.strip() for s in slot_str.split(",")]
            action = [action_name, slots]
        else:
            # utterance
            next_utterance = parts[2].strip()

    return intent, next_step, action, next_utterance


def compute_cds_em_and_ce(predictions, labels, convo_ids, turn_ids):
    """Adapted from ABCD. """
    expected, predicted = [], []
    intent_preds = []
    intent_labels = []

    next_step_preds = []
    next_step_labels = []

    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []

    utterance_preds = []
    utterance_labels = []

    num_of_action_turn = 0

    num_utt_turns = 0
    for pred, label in zip(predictions, labels):
        expected.append(label.strip())
        predicted.append(pred.strip())

        intent_label, next_step_label, action_value_label, utterance_label = parse_cds_prediction(label)
        intent_pred, next_step_pred, action_value_pred, utterance_pred = parse_cds_prediction(pred)

        intent_preds.append(intent_pred)
        intent_labels.append(intent_label)

        next_step_preds.append(next_step_pred)
        next_step_labels.append(next_step_label)

        if next_step_label == "action":
            num_of_action_turn += 1

            action_label, values_label = action_value_label
            values_label.sort()

            action_pred, values_pred = action_value_pred
            values_pred.sort()

            action_labels.append(action_label)
            value_labels.append(values_label)

            if len(values_pred) > len(values_label):
                values_pred = [v for v in values_label if v in values_pred]
            if len(values_pred) < len(values_label):
                values_pred.extend(["MISSING"] * (len(values_label) - len(values_pred)))


            action_preds.append(action_pred)
            value_preds.append(values_pred)
        else:
            # Mimic abcd
            action_preds.append(-1)
            value_preds.append(-1)
            action_labels.append(-1)
            value_labels.append(-1)

        if next_step_label == "respond":
            num_utt_turns += 1
            utterance_preds.append(utterance_pred)
            utterance_labels.append(utterance_label)
        else:
            # Needed for CE calculation
            utterance_labels.append(-1)
            utterance_preds.append(-1)

    num_turns = len(expected)

    # Intent
    intent_preds_array = np.array(intent_preds)
    intent_labels_array = np.array(intent_labels)
    intent_match = intent_labels_array == intent_preds_array
    intent_acc = sum(intent_match) / float(num_turns)

    # Next Step
    next_step_preds_array = np.array(next_step_preds)
    next_step_labels_array = np.array(next_step_labels)
    next_step_match = next_step_labels_array == next_step_preds_array
    next_step_acc = sum(next_step_match) / float(num_turns)

    # action

    action_labels_arrary = np.array(action_labels)
    action_preds_arrary = np.array(action_preds)
    action_match = action_labels_arrary == action_preds_arrary
    selector = action_labels_arrary != "-1"
    action_match = action_match * selector
    action_acc = sum(action_match) / float(num_of_action_turn)

    value_labels_arrary = np.array(value_labels)
    value_preds_arrary = np.array(value_preds)
    value_match = value_labels_arrary == value_preds_arrary
    value_match = value_match * selector
    value_acc = sum(value_match) / float(num_of_action_turn)

    joint_match = action_match & value_match
    joint_acc = sum(joint_match) / float(num_of_action_turn)

    utterance_labels_array = np.array(utterance_labels)
    utterance_preds_array = np.array(utterance_preds)
    utterance_match = utterance_labels_array == utterance_preds_array
    utt_selector = utterance_labels_array != "-1"
    utterance_match = utterance_match * utt_selector
    utterance_recall_1 = sum(utterance_match) / num_utt_turns

    # group by convo_ids
    unique_convo_ids = list(set(convo_ids))
    conversations = {}
    for uci in unique_convo_ids:
        turns, correctness = [], []
        row_id = 0
        for convo_id, turn_count in zip(convo_ids, turn_ids):
            if convo_id == uci:
                turns.append(turn_count)
                correct = False
                intent_right = intent_match[row_id]
                nextstep_right = next_step_match[row_id]

                if next_step_labels[row_id] == "respond":
                    if intent_right and nextstep_right and utterance_match[row_id]:
                        correct = True
                    else:
                        correct = False
                elif next_step_labels[row_id] == "action":
                    if intent_right and nextstep_right and joint_match[row_id]:
                        correct = True
                    else:
                        correct = False
                elif next_step_labels[row_id] == "end":
                    if intent_right and nextstep_right:
                        correct = True
                    else:
                        correct = False
                else:
                    raise ValueError()

                correctness.append(correct)
            row_id += 1

        # sort by turn_counts
        ordered = [cor for _, cor in sorted(zip(turns, correctness), key=lambda tc: tc[0])]
        conversations[uci] = ordered

    # count how many correct
    turn_score, turn_correct = 0, 0
    my_scores = []
    for convo_id, convo_correctness in conversations.items():
        current_score = 0
        convo_length = len(convo_correctness)
        # we use turn_id rather than the true turn_count since turn counts will skip numbers
        # when looping through the conversation due to skipping over customer utterances
        for turn_id in range(convo_length):
            num_remaining = convo_length - turn_id

            num_correct = 0
            # count up how many were predicted correctly
            while turn_id < convo_length and convo_correctness[turn_id]:
                num_correct += 1
                turn_id += 1

            if num_correct > 0:
                turn_correct += 1
            # normalize by the number of turns remaining
            turn_score += num_correct / num_remaining
            current_score += num_correct / num_remaining

        my_scores.append(current_score / convo_length)

    # normalize by total number of turns possible
    turn_acc = turn_correct / float(num_turns)
    final_score = turn_score / float(num_turns)

    full_result = {
        "Intent_Accuracy": round(intent_acc, 4),
        "Nextstep_Accuracy": round(next_step_acc, 4),
        "Action_Accuracy": round(action_acc, 4),
        "Value_Accuracy": round(value_acc, 4),
        "Joint_Accuracy": round(joint_acc, 4),
        "Recall_at_1": round(utterance_recall_1, 4),
        "Recall_at_5": "N/A",
        "Recall_at_10": "N/A",
        "Turn_Accuracy": round(turn_acc, 4),
        "Cascading_Score": round(final_score, 4),
    }

    return full_result

def parse_ast_prediction(prediction_str):
    match = re.match(r"(.*)\[(.*)]", prediction_str)
    if match:
        # action w/ value
        action_name = match.group(1).strip()
        slot_str = match.group(2)
        slot_str = slot_str.replace(";", ",")
        slots = [s.strip() for s in slot_str.split(",")]
        '''
        added by xiangyu
        '''
        for i in range(len(slots)):
            if slots[i].endswith(">") and not slots[i].startswith("<"):
                # add "<" to the beginning of the slot
                slots[i] = "<" + slots[i]
            if slots[i].startswith("<") and not slots[i].endswith(">"):
                # add ">" to the end of the slot
                slots[i] = slots[i] + ">"
    else:
        print(f"prediction_str: {prediction_str}")
        action_name = "MISSING"
        slots = ["MISSING"]

    return action_name, slots

'''
modified implementation with convo_ids and turn_ids
version: 1.0 
description: 
    1. markov chain prior * a + sequence score * (1 - a)
    2. only use the predicted actions as the final prediction (predicted actions may be wrong in format)
'''
def compute_ast_acc_metrics(predictions, labels, convo_ids, turn_ids, sequence_scores=None):
    # print("predictions: ", predictions)
    # print("labels: ", labels)


    from aalpy.utils import load_automaton_from_file

    # load an automaton
    automaton = load_automaton_from_file("./chainPrior/learned_mdp_8000.dot", automaton_type='mdp')
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

    '''
    state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}
    '''
    # print(f"state_mapping: {state_mapping}")

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

    # group predictions every 4
    new_predictions = []
    for i in range(0, len(predictions), 4):
        new_predictions.append(predictions[i:i+4])
        # new_predictions.append(predictions[i])
    
    new_sequence_scores = []
    for i in range(0, len(sequence_scores), 4):
        # new_sequence_scores.append(sequence_scores[i:i+4]/np.sum(sequence_scores[i:i+4]))
        new_sequence_scores.append(sequence_scores[i:i+4])

    
    previous_actions = ['init']
    current_convo_id = 999999
    new_new_predictions = []
    for new_pred, label1, new_sequence_score, convo_id1, turn_id1 in zip(new_predictions, labels, new_sequence_scores, convo_ids, turn_ids):
        # print("new_pred:", new_pred)
        # print("new_sequence_score:", new_sequence_score)
        # print()
        # print(f"new_pred[0]: {new_pred[0]}, new_pred[1]: {new_pred[1]}, new_pred[2]: {new_pred[2]}, new_pred[3]: {new_pred[3]}")
        action1 = new_pred[0].split(' ')[0].strip()
        action2 = new_pred[1].split(' ')[0].strip()
        action3 = new_pred[2].split(' ')[0].strip()
        action4 = new_pred[3].split(' ')[0].strip()

        # print(f"action1: {action1}, action2: {action2}, action3: {action3}, action4: {action4}")

        if convo_id1 != current_convo_id:
            previous_actions = ['init']
            current_convo_id = convo_id1

        try:
            rate1 = transition_mapping[(previous_actions[-1], action1)][1]
        except:
            # print(f"previous_actions[-1]: {previous_actions[-1]}, action1: {action1}")
            rate1 = 0.0
        try:
            rate2 = transition_mapping[(previous_actions[-1], action2)][1]
        except:
            # print(f"previous_actions[-1]: {previous_actions[-1]}, action2: {action2}")
            rate2 = 0.0
        try:
            rate3 = transition_mapping[(previous_actions[-1], action3)][1]
        except:
            # print(f"previous_actions[-1]: {previous_actions[-1]}, action3: {action3}")
            rate3 = 0.0
        try:
            rate4 = transition_mapping[(previous_actions[-1], action4)][1]
        except:
            # print(f"previous_actions[-1]: {previous_actions[-1]}, action4: {action4}")
            rate4 = 0.0

        rates = [rate1, rate2, rate3, rate4]
        rates = np.array(rates)
        # print(f"rates: {rates}")
        # print(f"new_sequence_score: {new_sequence_score}")
        # if model_args.chainPrior:
        # merge_scores = 0.7*new_sequence_score + 0.3*rates
        # else:
        merge_scores = new_sequence_score
        # merge_scores = rates
        # print(f"merge_scores: {merge_scores}")
        max_index = np.argmax(merge_scores)
        # print(f"max_index: {max_index}")
        # print(new_pred[max_index])
        # print(label1)
        # print()

        new_new_predictions.append(new_pred[max_index])

        previous_actions.append(label1.split(' ')[0].strip())

    """Adapted from ABCD. """
    # print("predictions:", predictions)
    # print("labels:", labels)
    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []
    
    # print("len(new_new_predictions): ", len(new_new_predictions))
    # print("len(labels): ", len(labels))
    for pred, label in zip(new_new_predictions, labels):
        pred = pred.split(";")[0]
        label = label.split(";")[0]
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

    # print("action_preds: ", action_preds)
    # print("action_labels: ", action_labels)
    # print("value_labels: ", value_labels)
    # print("convo_ids: ", convo_ids)
    # print("turn_ids: ", turn_ids)

    action_labels_arrary = np.array(action_labels, dtype=object)
    action_preds_arrary = np.array(action_preds, dtype=object)
    # print(f"action_labels_arrary: {action_labels_arrary}")
    # print(f"action_preds_arrary: {action_preds_arrary}")
    action_match = action_labels_arrary == action_preds_arrary
    # print(f"action_match: {action_match}")
    # print()
    action_acc = sum(action_match) / float(len(action_labels))

    value_labels_arrary = np.array(value_labels, dtype=object)
    value_preds_arrary = np.array(value_preds, dtype=object)
    # print(f"value_labels_arrary: {value_labels_arrary}")
    # print(f"value_preds_arrary: {value_preds_arrary}")
    value_match = value_labels_arrary == value_preds_arrary
    # print(f"value_match: {value_match}")
    value_acc = sum(value_match) / float(len(action_labels))

    joint_match = action_match & value_match
    joint_acc = sum(joint_match) / float(len(action_labels))

    # group by convo_ids
    unique_convo_ids = list(set(convo_ids))
    # print(f"unique_convo_ids: {unique_convo_ids}")
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
                # print(f"action_right: {action_right}, value_right: {value_right}")
                
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


'''
modified implementation with convo_ids and turn_ids
version: 2.0 
description: vanilla
'''
def compute_ast_acc_metrics_v2(predictions, labels, convo_ids, turn_ids, sequence_scores=None):
    from aalpy.utils import load_automaton_from_file

    # load an automaton
    automaton = load_automaton_from_file("/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/chainPrior/learned_mdp_8000.dot", automaton_type='mdp')
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

    print(f"state_mapping: {state_mapping}")

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

    # group predictions every 4
    new_predictions = []
    for i in range(0, len(predictions), 4):
        new_predictions.append(predictions[i:i+4])
        # new_predictions.append(predictions[i])
    
    new_sequence_scores = []
    for i in range(0, len(sequence_scores), 4):
        new_sequence_scores.append(sequence_scores[i:i+4]/np.sum(sequence_scores[i:i+4]))

    
    previous_actions = ['init']
    current_convo_id = 999999
    new_new_predictions = []
    for new_pred, label1, new_sequence_score, convo_id1, turn_id1 in zip(new_predictions, labels, new_sequence_scores, convo_ids, turn_ids):
        # print("new_pred:", new_pred)
        # print("new_sequence_score:", new_sequence_score)
        # print()
        # print(f"new_pred[0]: {new_pred[0]}, new_pred[1]: {new_pred[1]}, new_pred[2]: {new_pred[2]}, new_pred[3]: {new_pred[3]}")
        action1 = new_pred[0].split(' ')[0].strip()
        action2 = new_pred[1].split(' ')[0].strip()
        action3 = new_pred[2].split(' ')[0].strip()
        action4 = new_pred[3].split(' ')[0].strip()

        # print(f"action1: {action1}, action2: {action2}, action3: {action3}, action4: {action4}")

        if convo_id1 != current_convo_id:
            previous_actions = ['init']
            current_convo_id = convo_id1

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
        # print(f"rates: {rates}")
        # print(f"new_sequence_score: {new_sequence_score}")
        # if model_args.chainPrior:
        # merge_scores = 0.9*new_sequence_score + 0.1*rates
        # else:
        merge_scores = new_sequence_score
        # print(f"merge_scores: {merge_scores}")
        max_index = np.argmax(merge_scores)
        # print(f"max_index: {max_index}")
        # print(new_pred[max_index])
        # print(label1)
        # print()

        new_new_predictions.append(new_pred[max_index])

        previous_actions.append(label1.split(' ')[0].strip())

    """Adapted from ABCD. """
    # print("predictions:", predictions)
    # print("labels:", labels)
    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []
    
    print("len(new_new_predictions): ", len(new_new_predictions))
    print("len(labels): ", len(labels))
    for pred, label in zip(new_new_predictions, labels):
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

    # print("action_preds: ", action_preds)
    # print("action_labels: ", action_labels)
    # print("value_labels: ", value_labels)
    # print("convo_ids: ", convo_ids)
    # print("turn_ids: ", turn_ids)

    action_labels_arrary = np.array(action_labels, dtype=object)
    action_preds_arrary = np.array(action_preds, dtype=object)
    # print(f"action_labels_arrary: {action_labels_arrary}")
    # print(f"action_preds_arrary: {action_preds_arrary}")
    action_match = action_labels_arrary == action_preds_arrary
    # print(f"action_match: {action_match}")
    # print()
    action_acc = sum(action_match) / float(len(action_labels))

    value_labels_arrary = np.array(value_labels, dtype=object)
    value_preds_arrary = np.array(value_preds, dtype=object)
    # print(f"value_labels_arrary: {value_labels_arrary}")
    # print(f"value_preds_arrary: {value_preds_arrary}")
    value_match = value_labels_arrary == value_preds_arrary
    # print(f"value_match: {value_match}")
    value_acc = sum(value_match) / float(len(action_labels))

    joint_match = action_match & value_match
    joint_acc = sum(joint_match) / float(len(action_labels))

    # group by convo_ids
    unique_convo_ids = list(set(convo_ids))
    # print(f"unique_convo_ids: {unique_convo_ids}")
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
                # print(f"action_right: {action_right}, value_right: {value_right}")
                
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

'''
original implementation
'''
# def compute_ast_acc_metrics(predictions, labels):
#     """Adapted from ABCD. """
#     action_preds = []
#     action_labels = []

#     value_preds = []
#     value_labels = []

#     for pred, label in zip(predictions, labels):

#         action_label, values_label = parse_ast_prediction(label)
#         values_label.sort()
#         for value in values_label:
#             action_labels.append(action_label)
#             value_labels.append(value)

#         action_pred, values_pred = parse_ast_prediction(pred)
#         values_pred.sort()

#         if len(values_pred) > len(values_label):
#             values_pred = [v for v in values_label if v in values_pred]
#         if len(values_pred) < len(values_label):
#             values_pred.extend(["MISSING"] * (len(values_label) - len(values_pred)))

#         for value in values_pred:
#             action_preds.append(action_pred)
#             value_preds.append(value)

#     action_labels_arrary = np.array(action_labels)
#     action_preds_arrary = np.array(action_preds)
#     action_match = action_labels_arrary == action_preds_arrary
#     action_acc = sum(action_match) / float(len(action_labels))

#     value_labels_arrary = np.array(value_labels)
#     value_preds_arrary = np.array(value_preds)
#     value_match = value_labels_arrary == value_preds_arrary
#     value_acc = sum(value_match) / float(len(action_labels))

#     joint_match = action_match & value_match
#     joint_acc = sum(joint_match) / float(len(action_labels))

#     return {
#         "action": action_acc,
#         "value": value_acc,
#         "joint": joint_acc
#     }


def parse_workflow_string(workflow_str: str):
    workflow = []
    actions = workflow_str.split("; ")
    for action in actions:
        match = re.match(r"(.*)\[(.*)]", action)
        if match:
            # Has slots
            step_name = match.group(1).strip()
            slot_str = match.group(2)
            slot_str = slot_str.replace(";", ",")
            slots = [s.strip() for s in slot_str.split(",")]
        else:
            step_name = action.strip()
            slots = []

        workflow.append((step_name, slots))

    return workflow


def load_raw_test_dataset(file_path: Path, max_samples):
    convo_ids = []
    turn_counts = []
    with jsonlines.open(file_path) as reader:
        for sample in reader:
            convo_ids.append(sample["convo_id"])
            turn_counts.append(sample["turn_id"])
            if len(convo_ids) == max_samples:
                break
    return convo_ids, turn_counts


def create_compute_metric_fct(tokenizer, data_args, training_args, model_args):
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

    def parse_predictions(eval_preds):
        preds, labels = eval_preds
        decoded_predictions, decoded_labels = decode(preds, labels)
        return decoded_predictions, decoded_labels

    def compute_em_and_ce(eval_preds):
        predictions, labels = parse_predictions(eval_preds)
        predictions = [parse_workflow_string(w) for w in predictions]
        labels = [parse_workflow_string(w) for w in labels]
        return compute_metrics(labels, predictions, use_bert_score=training_args.use_bert_score)

    def compute_cds_metrics(eval_preds):
        predictions, labels = parse_predictions(eval_preds)
        # print("data_args.test_file", data_args.test_file)
        # print("data_args.max_predict_samples", data_args.max_predict_samples)
        convo_ids, turn_ids = load_raw_test_dataset(data_args.test_file, data_args.max_predict_samples)
        return compute_cds_em_and_ce(predictions, labels, convo_ids, turn_ids)

    def compute_ast_metrics(eval_preds, sequence_scores=None):
        predictions, labels = parse_predictions(eval_preds)
        is_eval = True if len(labels) == 3684 else False
        if is_eval:
            conv_ids, turn_ids = load_raw_test_dataset(data_args.validation_file, data_args.max_predict_samples)
        else:
            conv_ids, turn_ids = load_raw_test_dataset(data_args.test_file, data_args.max_predict_samples)
        print("predictions:", len(predictions))
        print("labels:", len(labels))
        print("conv_ids:", len(conv_ids))
        print("turn_ids:", len(turn_ids))
        print("sequence_scores:", len(sequence_scores))
        '''
        predictions: 200
        labels: 50
        conv_ids: 50
        turn_ids: 50
        sequence_scores: (200,)
        '''
        return compute_ast_acc_metrics(predictions, labels, conv_ids, turn_ids, sequence_scores)

    def no_metrics(eval_preds):
        # Evaluation will be done during post hf_training
        preds, labels = eval_preds
        decode(preds, labels)
        return {}

    if training_args.no_metrics:
        return no_metrics
    elif training_args.use_cds_metrics:
        return compute_cds_metrics
    elif training_args.use_ast_metrics:
        return compute_ast_metrics
    else:
        return compute_em_and_ce
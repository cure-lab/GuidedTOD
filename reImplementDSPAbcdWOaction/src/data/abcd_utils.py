"""All processing functions are adapted from https://github.com/asappresearch/abcd/blob/master/utils/process.py
for fair comparison for the AST and CDS tasks.
"""
import json
from pathlib import Path
import random
from typing import List, Dict

from tqdm import tqdm as progress_bar

# set the seed
random.seed(42)

def prepare_action_labels(ontology):
    action_list = []
    for section, buttons in ontology["actions"].items():
        actions = buttons.keys()
        action_list.extend(actions)
    return {action: idx for idx, action in enumerate(action_list)}


def prepare_value_labels(ontology):
    value_list = []
    for category, values in ontology["values"]["enumerable"].items():
        for val in values:
            if val not in value_list:
                value_list.append(val.lower())
    return {slotval: idx for idx, slotval in enumerate(value_list)}


def _read_json_file(raw_data_path: Path, file_name: str):
    file_path = raw_data_path / file_name
    with file_path.open() as f:
        data = json.load(f)

    return data

def read_abcd_raw_data(raw_data_path: Path):
    return _read_json_file(raw_data_path, "abcd_v1.1.json")


def read_abcd_guidelines(raw_data_path: Path):
    return _read_json_file(raw_data_path, "guidelines.json")


def read_abcd_ontology(raw_data_path: Path):
    return _read_json_file(raw_data_path, "ontology.json")


def read_utterances_file(raw_data_path: Path):
    return _read_json_file(raw_data_path, "utterances.json")


def prepare_labels_for_ast(raw_data_path: Path):
    ontology = read_abcd_ontology(raw_data_path)

    non_enumerable = ontology["values"]["non_enumerable"]
    enumerable = {}
    for category, values in ontology["values"]["enumerable"].items():
        enumerable[category] = [val.lower() for val in values]

    mappers = {"value": prepare_value_labels(ontology), "action": prepare_action_labels(ontology)}

    # Break down the slot values by action
    value_by_action = {}
    for section, actions in ontology["actions"].items():
        for action, targets in actions.items():
            value_by_action[action] = targets

    return non_enumerable, enumerable, mappers, value_by_action


def ast_value_to_id(_context, value, potential_vals, enumerable):
    for option in potential_vals:
        if option in enumerable:  # just look it up
            if value in enumerable[option]:
                # We need to return the exact value
                # potential_vals.pop(potential_vals.index(option))
                return value
        else:
            entity = f"<{option}>"  # calculate location in the context
            if entity in _context:
                # We need to return the entity
                return entity

    return value


def prepare_intent_labels(ontology):
    intent_list = []
    for flow, subflows in ontology["intents"]["subflows"].items():
        intent_list.extend(subflows)
    return {intent: idx for idx, intent in enumerate(intent_list)}


def prepare_nextstep_labels(ontology):
    nextstep_list = ontology["next_steps"]
    return {nextstep: idx for idx, nextstep in enumerate(nextstep_list)}


def prepare_labels_for_cds(raw_data_path: Path):
    ontology = read_abcd_ontology(raw_data_path)

    non_enumerable = ontology["values"]["non_enumerable"]
    enumerable = {}
    for category, values in ontology["values"]["enumerable"].items():
        enumerable[category] = [val.lower() for val in values]

    mappers = {
        "value": prepare_value_labels(ontology),
        "action": prepare_action_labels(ontology),
        "intent": prepare_intent_labels(ontology),
        "nextstep": prepare_nextstep_labels(ontology),
    }  # utterance is ranking, so not needed

    # Break down the slot values by action
    value_by_action = {}
    for section, actions in ontology["actions"].items():
        for action, targets in actions.items():
            value_by_action[action] = targets

    return non_enumerable, enumerable, mappers, value_by_action


def collect_one_example(dialog_history, targets, support_items, enumerable, mappers, utterances):
    def value_to_id(_context, value, potential_vals):
        for option in potential_vals:
            if option in enumerable:  # just look it up
                if value in enumerable[option]:
                    return value
            else:
                entity = f"<{option}>"  # calculate location in the context
                if entity in _context:
                    return entity

        return value

    def action_to_id(action):
        return mappers["action"][action]

    intent, nextstep, action, _, utt_id = targets
    take_action_target = ["none", ["none"]]
    utt_candidates = ""
    target_utterance = "none"

    if nextstep == "take_action":
        value, potential_vals, convo_id, turn_id = support_items
        if value != "not applicable":
            parsed_values = []
            for v in value:
                value_id = value_to_id(dialog_history, v, potential_vals)
                parsed_values.append(value_id)
        else:
            parsed_values = ["none"]
        take_action_target = [action, parsed_values]
        nextstep_target = "action"

    elif nextstep == "retrieve_utterance":
        candidates, convo_id, turn_id = support_items
        target_utt_id = candidates[utt_id]
        target_utterance = utterances[target_utt_id]
        real_candidates = [utterances[u] for u in candidates]
        utt_candidates = real_candidates
        nextstep_target = "respond"

    elif nextstep == "end_conversation":
        convo_id, turn_id = support_items
        nextstep_target = "end"
    else:
        raise ValueError()

    return {
        "context": [t.split("|")[1] for t in dialog_history],
        "intent": intent,
        "next_step": nextstep_target,
        "take_action_target": take_action_target,
        "target_utterance": target_utterance,
        "candidates": utt_candidates,
        "convo_id": convo_id,
        "turn_id": turn_id,
    }


def collect_examples(context, targets, convo_id, turn_id, value_by_action, enumerable, mappers, utterances):
    _, _, action, values, _ = targets
    potential_vals = value_by_action[action]

    if len(potential_vals) > 0:  # just skip if action does not require inputs
        return collect_one_example(
            context, targets, (values, potential_vals, convo_id, turn_id), enumerable, mappers, utterances
        )
    else:
        return collect_one_example(
            context, targets, ("not applicable", potential_vals, convo_id, turn_id), enumerable, mappers, utterances
        )


def parse_abcd_dataset_for_cds(raw_dat_path: Path, data: List):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_cds(raw_dat_path)
    utterances = read_utterances_file(raw_dat_path)

    parsed_samples = []

    for sample in progress_bar(data, total=len(data)):
        so_far = []
        for turn in sample["delexed"]:
            speaker, text = turn["speaker"], turn["text"]
            utterance = f"{speaker}|{text}"

            if speaker == "agent":
                context = so_far.copy()
                support_items = turn["candidates"], sample["convo_id"], turn["turn_count"]
                parsed_samples.append(
                    collect_one_example(context, turn["targets"], support_items, enumerable, mappers, utterances)
                )
                so_far.append(utterance)
            elif speaker == "action":
                context = so_far.copy()
                parsed_samples.append(
                    collect_examples(
                        context,
                        turn["targets"],
                        sample["convo_id"],
                        turn["turn_count"],
                        value_by_action,
                        enumerable,
                        mappers,
                        utterances,
                    )
                )
                so_far.append(utterance)
            else:
                so_far.append(utterance)

        context = so_far.copy()  # the entire conversation
        end_targets = turn["targets"].copy()  # last turn after then end of the loop
        end_targets[1] = "end_conversation"
        end_targets[4] = -1
        support_items = sample["convo_id"], turn["turn_count"]
        parsed_samples.append(
            collect_one_example(context, end_targets, support_items, enumerable, mappers, utterances)
        )  # end

    return parsed_samples

# demo data in the test set
'''
{
    "convo_id": 4989, 
    "scenario": {
        "personal": {
            "customer_name": "chloe zhang", 
            "member_level": "gold", 
            "phone": "(741) 062-3341"
        }, 
        "order": {
            "street_address": "4247 lincoln ave", 
            "full_address": "4247 lincoln ave  newark, ca 82636", 
            "city": "newark", 
            "state": "ca", 
            "zip_code": "82636"
        }, 
        "product": {
            "names": [], 
            "amounts": []
        }, 
        "flow": "storewide_query", 
        "subflow": "timing_4"
    }, 
    "original": [["agent", "Hello. How can i help you today?"], ["customer", "Hi.  My name is Chloe Zhang.  I am curious as to when my promo code expires."], ["customer", "Would you be able to tell me?"], ["agent", "Yes let me look into this im sure we can find a solution"], ["action", "Searching the FAQ pages ..."], ["action", "System Action: search timing"], ["agent", "Okay I've looked into this for you"], ["agent", "Promo codes will expire 7 days after they are issued."], ["customer", "Ok."], ["agent", "Is there anything else I can help you with?"], ["customer", "No that is all for today.  Thank you."]], 
    "delexed": [
        {"speaker": "agent", "text": "hello. how can i help you today?", "turn_count": 1, "targets": ["timing", "retrieve_utterance", null, [], 3], "candidates": [15881, 13841, 35741, 7, 38413, 26432, 10617, 14627, 38914, 8954, 8856, 27786, 39665, 13325, 21311, 23767, 7728, 29573, 38379, 25517, 15472, 13884, 744, 15918, 20664, 2701, 39980, 5887, 33792, 36644, 3179, 31739, 34529, 27109, 7411, 17427, 16276, 38944, 20623, 16129, 22893, 28544, 2991, 32412, 37959, 30323, 14448, 18326, 30185, 36979, 13706, 38728, 15878, 27066, 34849, 27036, 25027, 18987, 26022, 13384, 6265, 3999, 9465, 8394, 18242, 39924, 9536, 32551, 20939, 14493, 28303, 29053, 1132, 11268, 34535, 33239, 6361, 3212, 34906, 8874, 3577, 41, 18049, 27214, 25343, 12402, 1801, 26279, 29900, 7348, 10966, 19871, 1260, 39304, 258, 484, 2563, 14906, 33042, 568]}, 
        {"speaker": "customer", "text": "hi.  my name is chloe zhang.  i am curious as to when my promo code expires.", "turn_count": 2, "targets": ["timing", null, null, [], -1], "candidates": []}, 
        {"speaker": "customer", "text": "would you be able to tell me?", "turn_count": 3, "targets": ["timing", null, null, [], -1], "candidates": []}, 
        {"speaker": "agent", "text": "yes let me look into this im sure we can find a solution", "turn_count": 4, "targets": ["timing", "retrieve_utterance", null, [], 14], "candidates": [5074, 4145, 20655, 4069, 21679, 32688, 32756, 12601, 26069, 10592, 29686, 32144, 26864, 24261, 66642, 2039, 18989, 23649, 39816, 7294, 16948, 17469, 5964, 24797, 10824, 38568, 28844, 22228, 9902, 6831, 516, 10282, 31648, 26194, 35023, 16252, 23148, 324, 9811, 15223, 15625, 9523, 15062, 37665, 39047, 25515, 28718, 599, 16741, 15996, 27649, 4615, 19346, 21665, 19012, 10231, 30932, 21845, 21947, 6756, 26535, 23362, 8809, 2544, 27105, 34254, 38433, 19624, 21895, 16837, 1882, 4211, 30806, 2359, 5599, 35296, 32462, 4911, 27308, 37339, 18404, 31371, 1651, 10119, 37784, 14381, 33542, 19631, 4649, 6967, 31031, 18167, 23439, 36896, 4648, 9747, 12823, 13522, 14899, 32903]}, 
        {"speaker": "action", "text": "searching the faq pages ...", "turn_count": 5, "targets": ["timing", "take_action", "search-faq", [], -1], "candidates": []}, 
        {"speaker": "action", "text": "system action: search timing", "turn_count": 6, "targets": ["timing", "take_action", "search-timing", [], -1], "candidates": []}, 
        {"speaker": "agent", "text": "okay i've looked into this for you", "turn_count": 7, "targets": ["timing", "retrieve_utterance", null, [], 76], "candidates": [33761, 30857, 13328, 20948, 13179, 32568, 100, 33319, 39552, 19402, 13537, 29557, 34040, 16445, 12873, 19209, 33787, 13536, 39492, 20928, 2276, 13038, 19249, 12563, 18435, 15064, 18119, 39083, 859, 18532, 24539, 18122, 39252, 26473, 11130, 31179, 28451, 15091, 12197, 21264, 18279, 20785, 17199, 23168, 1496, 26604, 15785, 13946, 4948, 24508, 13789, 21351, 19203, 37611, 5366, 10138, 23624, 39380, 14766, 6090, 15641, 5793, 27847, 13484, 20725, 4223, 29739, 28030, 32055, 39331, 37676, 7655, 25287, 11049, 32570, 19547, 66643, 36450, 32723, 3227, 36246, 19001, 38145, 6001, 19660, 22771, 6588, 12938, 7178, 14409, 19773, 8824, 39533, 725, 7267, 36454, 12406, 6495, 11465, 24533]}, 
        {"speaker": "agent", "text": "promo codes will expire 7 days after they are issued.", "turn_count": 8, "targets": ["timing", "retrieve_utterance", null, [], 71], "candidates": [21575, 35646, 26484, 5135, 14699, 33519, 13801, 28827, 255, 3609, 1413, 39311, 19272, 11747, 17934, 4410, 7253, 1613, 18065, 4547, 23330, 4109, 20329, 11066, 25016, 14610, 35022, 23953, 24296, 35395, 29991, 28176, 12262, 28562, 1903, 4902, 13056, 37433, 34376, 14536, 36397, 22206, 15308, 14164, 37362, 29950, 19249, 6483, 21338, 24393, 20953, 11738, 36491, 21468, 31258, 3301, 23042, 13117, 17556, 35539, 31587, 12560, 30644, 23118, 35545, 26865, 35313, 36342, 30515, 11966, 35249, 66644, 11551, 30781, 24832, 15167, 22409, 35148, 22124, 18980, 16861, 31247, 15775, 20773, 9014, 7539, 6152, 18173, 15974, 24664, 32557, 27026, 9234, 29331, 2066, 13938, 7032, 14863, 32166, 33572]}, 
        {"speaker": "customer", "text": "ok.", "turn_count": 9, "targets": ["timing", null, null, [], -1], "candidates": []}, 
        {"speaker": "agent", "text": "is there anything else i can help you with?", "turn_count": 10, "targets": ["timing", "retrieve_utterance", null, [], 15], "candidates": [32329, 17913, 18480, 15165, 20893, 721, 6968, 9334, 15651, 38187, 20165, 3760, 23343, 7214, 20154, 407, 31545, 19034, 36187, 21410, 17688, 37908, 12281, 4061, 39057, 33530, 23752, 29721, 2941, 1934, 1959, 9008, 35462, 24874, 30907, 21649, 3188, 21440, 29344, 27922, 13120, 30753, 23456, 13860, 6181, 14817, 25983, 20152, 21923, 38631, 35698, 334, 19379, 23334, 36051, 322, 3088, 14608, 20071, 4210, 36513, 38256, 22463, 17968, 564, 1050, 36745, 4106, 30573, 18453, 8585, 16765, 25733, 15507, 5677, 29175, 6558, 13348, 7430, 17013, 13372, 34953, 29804, 30871, 29169, 30258, 633, 31073, 7263, 29728, 3506, 10525, 39815, 23489, 18953, 26006, 726, 10135, 12356, 17672]}, 
        {"speaker": "customer", "text": "no that is all for today.  thank you.", "turn_count": 11, "targets": ["timing", null, null, [], -1], "candidates": []}
    ]
}
'''

'''
add action to context
'''
def parse_abcd_dataset_for_ast_w_action(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        context = []

        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                context.append(target_action + " " + str(target_values))
                target_action = sample['scenario']['flow'] + ":" + target_action
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"], "flow": sample["scenario"]["flow"]})
            else:
                context.append(turn["text"])
    return parsed_samples

'''
add most possible action, which is extracted from the chainprior, to context
Actions are ordered by the frequency of the transition, and select the most possible actions, whose sum of the frequency is greater than 0.8
'''
def parse_abcd_dataset_for_ast_w_mostpossible_chain_action(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    from aalpy.utils import load_automaton_from_file
    import numpy as np

    '''
    1. add from the global perspective
    '''
    automaton_global = load_automaton_from_file("./chainPrior/learned_mdp_8000.dot", automaton_type='mdp')
    automaton_global = str(automaton_global)

    automaton_splits = automaton_global.split('\n')
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

    all_possible_actions = ['init','pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

    transition_mapping_global = {}
    chain_action_candidates_global = {}
    for transition in automaton_transitions:
        transition_split = transition.split('->')
        source_state = transition_split[0].strip()
        target_state = transition_split[1].strip().split(' ')[0]
        transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
        transition_action = transition_label.split(':')[0]
        transition_freq = float(transition_label.split(':')[1])
        transition_mapping_global[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)
        if state_mapping[source_state] in all_possible_actions:
            if state_mapping[source_state] not in chain_action_candidates_global:
                chain_action_candidates_global[state_mapping[source_state]] = []
            if state_mapping[target_state] in all_possible_actions and transition_freq > 0.0:
                chain_action_candidates_global[state_mapping[source_state]].append((state_mapping[target_state], transition_freq))
                # chain_action_candidates[state_mapping[source_state]].append(state_mapping[target_state])
            else:
                pass
        else:
            continue
    
    for paction in all_possible_actions:
        if paction not in chain_action_candidates_global:
            chain_action_candidates_global[paction] = []
        if chain_action_candidates_global[paction] == []:
            chain_action_candidates_global[paction] = all_possible_actions

    # order the actions by the frequency of the transition, and select the most possible actions, whose sum of all the selected frequency is greater than 0.8
    for key, value in chain_action_candidates_global.items():
        # print(f"key: {key}, value: {value}")
        if not isinstance(value[0], tuple):
            continue
        value.sort(key=lambda x: x[1], reverse=True)
        selected_actions = []
        freq_sum = 0.0
        for action, freq in value:
            if freq_sum + freq <= 0.9:
                selected_actions.append([action, freq])
                freq_sum += freq
            elif freq_sum + freq > 0.9 and freq_sum < 0.9:
                selected_actions.append([action, freq])
                freq_sum += freq
            else:
                break
        chain_action_candidates_global[key] = selected_actions

    '''
    2. add from the perspective of subflows
    '''
    all_flow_name = [
        "storewide_query",
        "subscription_inquiry",
        "single_item_query",
        "troubleshoot_site",
        "purchase_dispute",
        "account_access",
        "shipping_issue",
        "order_issue",
        "product_defect",
        "manage_account"
    ]
    transition_mapping_flow = {}
    chain_action_candidates_flow = {}
    for i in range(len(all_flow_name)):
        flow_name = all_flow_name[i]
        chainedPriorpath = "./chainPrior/learned_mdp_abcd_flow_all_" + flow_name + ".dot"

        automaton = load_automaton_from_file(chainedPriorpath, automaton_type='mdp')
        automaton = str(automaton)
        automaton_splits = automaton.split('\n')
        automaton_states = automaton_splits[1:33]
        automaton_transitions = automaton_splits[33:-4]

        state_mapping = {}
        for state in automaton_states:
            state_name = state.split(' ')[0]
            state_label = state.split('[label="')[1].split('"];')[0]
            state_mapping[state_name] = state_label

        '''
        state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}
        '''
        transition_mapping = {}
        chain_action_candidates = {}
        for transition in automaton_transitions:
            transition_split = transition.split('->')
            source_state = transition_split[0].strip()
            target_state = transition_split[1].strip().split(' ')[0]
            transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
            transition_action = transition_label.split(':')[0]
            transition_freq = float(transition_label.split(':')[1])
            transition_mapping[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)

            if state_mapping[source_state] in all_possible_actions:
                # print(f"state_mapping[source_state]: ", state_mapping[source_state])
                if state_mapping[source_state] not in chain_action_candidates:
                    chain_action_candidates[state_mapping[source_state]] = []
                if state_mapping[target_state] in all_possible_actions and transition_freq > 0.0 and state_mapping[target_state] != 'init':
                    chain_action_candidates[state_mapping[source_state]].append((state_mapping[target_state], transition_freq))
                    # chain_action_candidates[state_mapping[source_state]].append(state_mapping[target_state])
                else:
                    pass
            else:
                continue

        transition_mapping_flow[flow_name] = transition_mapping
        # print(transition_mapping)
        # print()
        chain_action_candidates_flow[flow_name] = chain_action_candidates

    # for key, value in chain_action_candidates_flow.items():
    #     # print(f"keys: {key}")
    #     chain_action_candidates = value
    #     for paction in all_possible_actions:
    #         if paction not in chain_action_candidates:
    #             chain_action_candidates[paction] = []
    #         if chain_action_candidates[paction] == []:
    #             chain_action_candidates[paction] = all_possible_actions
    #     chain_action_candidates_flow[key] = chain_action_candidates

    for key, value in chain_action_candidates_flow.items():
        chain_action_candidates = value
        for key_1, value_1 in chain_action_candidates.items():
            print(f"key: {key_1}, value: {value_1}")
            if len(value_1) == 0:
                continue
            if not isinstance(value_1[0], tuple):
                continue
            value_1.sort(key=lambda x: x[1], reverse=True)
            selected_actions = []
            freq_sum = 0.0
            for action, freq in value_1:
                if freq_sum + freq <= 0.9:
                    selected_actions.append([action, freq])
                    freq_sum += freq
                elif freq_sum + freq > 0.9 and freq_sum < 0.9:
                    selected_actions.append([action, freq])
                    freq_sum += freq
                else:
                    break
            chain_action_candidates[key_1] = selected_actions
            print(f"key: {key_1}, value: {selected_actions}")
        
        chain_action_candidates_flow[key] = chain_action_candidates

    # for key, value in chain_action_candidates_flow.items():
    #     print(f"flow: {key}")
    #     chain_action_candidates = value
    #     for key_1, value_1 in chain_action_candidates.items():
    #         print(f"key: {key_1}, value: {value_1}")
    #     print()

    # from s0 to s31, if some pair of states are not in the transition_mapping, then the frequency is 0
    # for i in range(32):
    #     for j in range(32):
    #         if (state_mapping[f's{i}'], state_mapping[f's{j}']) not in transition_mapping:
    #             transition_mapping[(state_mapping[f's{i}'], state_mapping[f's{j}'])] = ('unknown', 0.0)

    # print(f"transition_mapping: {transition_mapping}")
    # print(f"chain_action_candidates: {chain_action_candidates}")
    # for key, value in chain_action_candidates_global.items():
    #     print(f"key: {key}, value: {value}")


    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        context = []
        # state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}

        # get actions in this sample
        actions = []
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]
                actions.append(target_action)
        
        # add all actions to actions_for_context, randomly select 4 actions from all_possible_actions apart from actions to actions_for_context
        # actions_for_context = []
        # for action in actions:
        #     actions_for_context.append(action)
        # for i in range(4):
        #     action = random.choice(all_possible_actions)
        #     while action in actions_for_context:
        #         action = random.choice(all_possible_actions)
        #     actions_for_context.append(action)
        
        # # shuffle actions_for_context
        # random.shuffle(actions_for_context)

        # replace the above functions by the following:
        actions_for_context = []

        last_action = "init"
        for idx, turn in enumerate(sample["delexed"]):
            flow_name = sample['scenario']['flow']
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                context.append(target_action + " " + str(target_values))
                # context.append("possible actions: " + '[' + ', '.join(actions_for_context) + ']')
                if last_action == "init":
                    '''
                        action_rate:  {
                        "pull-up-account": 0.6694050286283296,
                        "search-faq": 0.2045058501369181,
                        "enter-details": 0.006596962907642519,
                        "offer-refund": 0.0011202389843166542,
                        "verify-identity": 0.017301468757779437,
                        "try-again": 0.035225292506845904,
                        "validate-purchase": 0.0043564849390092105,
                        "notify-team": 0.01157580283793876,
                        "log-out-in": 0.032984814538212594,
                        "record-reason": 0.003983071944236993,
                        "make-purchase": 0.0008712969878018422,
                        "instructions": 0.0028628329599203386,
                        "shipping-status": 0.0017425939756036844,
                        "membership": 0.0036096589494647746,
                        "ask-the-oracle": 0.0008712969878018422,
                        "subscription-status": 0.0013691809808314662,
                        "update-account": 0.0004978839930296241,
                        "make-password": 0.00024894199651481205,
                        "update-order": 0.0003734129947722181,
                        "promo-code": 0.0003734129947722181,
                        "send-link": 0.00012447099825740602
                        }
                    '''
                    # actions_for_context = [_ for _ in all_possible_actions]
                    # actions_for_context = ["0.67*pull-up-account", "0.20*search-faq", "0.04*try-again"]
                    actions_for_context = [f"{action[1]:.2f}*{action[0]}" for action in chain_action_candidates_flow[flow_name][last_action]]
                    last_action = target_action
                else:
                    # actions_for_context = chain_action_candidates[last_action]
                    actions_for_context = [f"{action[1]:.2f}*{action[0]}" for action in chain_action_candidates_flow[flow_name][last_action]]
                    last_action = target_action
                
                context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                target_action = sample['scenario']['flow'] + ":" + target_action
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"], "flow": sample["scenario"]["flow"]})
            else:
                if turn["text"][-1] not in ['.', '?', '!']:
                    turn['text'] = turn['text'] + '.'
                context.append(turn["text"])
    return parsed_samples

'''
add most possible action, which is extracted from the chainprior, to context
Actions are ordered by the frequency of the transition, and select the most possible actions, whose sum of the frequency is greater than 0.8
The sentences are augmented except for the action+value phrases
'''
def parse_abcd_dataset_for_ast_w_mostpossible_chain_action_aug(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    from aalpy.utils import load_automaton_from_file

    automaton = load_automaton_from_file("./chainPrior/learned_mdp_8000.dot", automaton_type='mdp')
    automaton = str(automaton)

    automaton_splits = automaton.split('\n')
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

    all_possible_actions = ['pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

    transition_mapping = {}
    chain_action_candidates = {}
    for transition in automaton_transitions:
        transition_split = transition.split('->')
        source_state = transition_split[0].strip()
        target_state = transition_split[1].strip().split(' ')[0]
        transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
        transition_action = transition_label.split(':')[0]
        transition_freq = float(transition_label.split(':')[1])
        transition_mapping[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)
        if state_mapping[source_state] in all_possible_actions:
            if state_mapping[source_state] not in chain_action_candidates:
                chain_action_candidates[state_mapping[source_state]] = []
            if state_mapping[target_state] in all_possible_actions and transition_freq > 0.0:
                chain_action_candidates[state_mapping[source_state]].append((state_mapping[target_state], transition_freq))
                # chain_action_candidates[state_mapping[source_state]].append(state_mapping[target_state])
            else:
                pass
        else:
            continue
    
    for paction in all_possible_actions:
        if paction not in chain_action_candidates:
            chain_action_candidates[paction] = []
        if chain_action_candidates[paction] == []:
            chain_action_candidates[paction] = all_possible_actions

    def augSentence(sentence):
        PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
        PUNC_RATIO = 0.3

        # Insert punction words into a given sentence with the given ratio "punc_ratio"
        def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
            words = sentence.split(' ')
            new_line = []
            q = random.randint(1, int(punc_ratio * len(words) + 1))
            qs = random.sample(range(0, len(words)), q)

            for j, word in enumerate(words):
                if j in qs:
                    new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
                    new_line.append(word)
                else:
                    new_line.append(word)
            new_line = ' '.join(new_line)
            return new_line

        sentence_aug = insert_punctuation_marks(sentence)
        return sentence_aug

    # order the actions by the frequency of the transition, and select the most possible actions, whose sum of all the selected frequency is greater than 0.8
    for key, value in chain_action_candidates.items():
        # print(f"key: {key}, value: {value}")
        if not isinstance(value[0], tuple):
            continue
        value.sort(key=lambda x: x[1], reverse=True)
        selected_actions = []
        freq_sum = 0.0
        for action, freq in value:
            if freq_sum + freq <= 0.8:
                selected_actions.append(action)
                freq_sum += freq
            elif freq_sum + freq > 0.8 and freq_sum < 0.8:
                selected_actions.append(action)
                freq_sum += freq
            else:
                break
        chain_action_candidates[key] = selected_actions

    # from s0 to s31, if some pair of states are not in the transition_mapping, then the frequency is 0
    # for i in range(32):
    #     for j in range(32):
    #         if (state_mapping[f's{i}'], state_mapping[f's{j}']) not in transition_mapping:
    #             transition_mapping[(state_mapping[f's{i}'], state_mapping[f's{j}'])] = ('unknown', 0.0)

    # print(f"transition_mapping: {transition_mapping}")
    # print(f"chain_action_candidates: {chain_action_candidates}")
    for key, value in chain_action_candidates.items():
        print(f"key: {key}, value: {value}")


    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    aug_candidate_actions = ["search-timing", 
                             "search-policy", 
                             "subscription-status", 
                             "instructions", 
                             "try-again", 
                             "shipping-status", 
                             "search-membership", 
                             "search-pricing"]

    for sample in data:
        context = []
        # state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}

        # get actions in this sample
        actions = []
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]
                actions.append(target_action)
        
        # add all actions to actions_for_context, randomly select 4 actions from all_possible_actions apart from actions to actions_for_context
        # actions_for_context = []
        # for action in actions:
        #     actions_for_context.append(action)
        # for i in range(4):
        #     action = random.choice(all_possible_actions)
        #     while action in actions_for_context:
        #         action = random.choice(all_possible_actions)
        #     actions_for_context.append(action)
        
        # # shuffle actions_for_context
        # random.shuffle(actions_for_context)

        # replace the above functions by the following:
        actions_for_context = []

        last_action = "init"
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                context.append(target_action + " " + str(target_values))
                # context.append("possible actions: " + '[' + ', '.join(actions_for_context) + ']')
                if last_action == "init":
                    # actions_for_context = [_ for _ in all_possible_actions]
                    actions_for_context = []
                    last_action = target_action
                else:
                    actions_for_context = chain_action_candidates[last_action]
                    last_action = target_action
                
                if target_action in aug_candidate_actions:
                    # context.append(augSentence(turn["text"]))
                    for aug_iter in range(4):
                        context__ = [c for c in context_[:]]
                        for sentence_iter in range(len(context__)):
                            if context__[sentence_iter].split(' ')[0] in all_possible_actions:
                                pass
                            else:
                                context__[sentence_iter] = augSentence(context__[sentence_iter])

                        context__.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                        parsed_samples.append({"context": context__, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
                    context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                    parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
                else:
                    context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                    parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
            else:
                if turn["text"][-1] not in ['.', '?', '!']:
                    turn['text'] = turn['text'] + '.'
                
                context.append(turn["text"])
    return parsed_samples

'''
add possible action, which is extracted from the chainprior, to context
'''
def parse_abcd_dataset_for_ast_w_possible_chain_action(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    from aalpy.utils import load_automaton_from_file

    automaton = load_automaton_from_file("./chainPrior/learned_mdp_8000.dot", automaton_type='mdp')
    automaton = str(automaton)

    automaton_splits = automaton.split('\n')
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

    all_possible_actions = ['pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

    transition_mapping = {}
    chain_action_candidates = {}
    for transition in automaton_transitions:
        transition_split = transition.split('->')
        source_state = transition_split[0].strip()
        target_state = transition_split[1].strip().split(' ')[0]
        transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
        transition_action = transition_label.split(':')[0]
        transition_freq = float(transition_label.split(':')[1])
        transition_mapping[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)
        if state_mapping[source_state] in all_possible_actions:
            if state_mapping[source_state] not in chain_action_candidates:
                chain_action_candidates[state_mapping[source_state]] = []
            if state_mapping[target_state] in all_possible_actions and transition_freq > 0.0:
                chain_action_candidates[state_mapping[source_state]].append((state_mapping[target_state], transition_freq))
                # chain_action_candidates[state_mapping[source_state]].append(state_mapping[target_state])
            else:
                pass
        else:
            continue
    
    for paction in all_possible_actions:
        if paction not in chain_action_candidates:
            chain_action_candidates[paction] = []
        if chain_action_candidates[paction] == []:
            chain_action_candidates[paction] = all_possible_actions

    # from s0 to s31, if some pair of states are not in the transition_mapping, then the frequency is 0
    # for i in range(32):
    #     for j in range(32):
    #         if (state_mapping[f's{i}'], state_mapping[f's{j}']) not in transition_mapping:
    #             transition_mapping[(state_mapping[f's{i}'], state_mapping[f's{j}'])] = ('unknown', 0.0)

    # print(f"transition_mapping: {transition_mapping}")
    # print(f"chain_action_candidates: {chain_action_candidates}")
    for key, value in chain_action_candidates.items():
        print(f"key: {key}, value: {value}")


    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        context = []
        # state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}

        # get actions in this sample
        actions = []
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]
                actions.append(target_action)
        
        # add all actions to actions_for_context, randomly select 4 actions from all_possible_actions apart from actions to actions_for_context
        # actions_for_context = []
        # for action in actions:
        #     actions_for_context.append(action)
        # for i in range(4):
        #     action = random.choice(all_possible_actions)
        #     while action in actions_for_context:
        #         action = random.choice(all_possible_actions)
        #     actions_for_context.append(action)
        
        # # shuffle actions_for_context
        # random.shuffle(actions_for_context)

        # replace the above functions by the following:
        actions_for_context = []

        last_action = "init"
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                # context.append(target_action + " " + str(target_values))
                # context.append("possible actions: " + '[' + ', '.join(actions_for_context) + ']')
                if last_action == "init":
                    actions_for_context = [_ for _ in all_possible_actions]
                    last_action = target_action
                else:
                    actions_for_context = chain_action_candidates[last_action]
                    last_action = target_action

                context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
            else:
                if turn["text"][-1] not in ['.', '?', '!']:
                    turn['text'] = turn['text'] + '.'
                context.append(turn["text"])
    return parsed_samples


'''
add possible action to context
'''
def parse_abcd_dataset_for_ast_w_possible_action(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        context = []
        # state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}
        all_possible_actions = ['pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

        # get actions in this sample
        actions = []
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]
                actions.append(target_action)
        
        # add all actions to actions_for_context, randomly select 4 actions from all_possible_actions apart from actions to actions_for_context
        actions_for_context = []
        for action in actions:
            actions_for_context.append(action)
        for i in range(4):
            action = random.choice(all_possible_actions)
            while action in actions_for_context:
                action = random.choice(all_possible_actions)
            actions_for_context.append(action)
        
        # shuffle actions_for_context
        random.shuffle(actions_for_context)

        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                # context.append(target_action + " " + str(target_values))
                # context.append("possible actions: " + '[' + ', '.join(actions_for_context) + ']')
                context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
            else:
                if turn["text"][-1] not in ['.', '?', '!']:
                    turn['text'] = turn['text'] + '.'
                context.append(turn["text"])
    return parsed_samples

'''
add possible action to context (upper bound)
'''
def parse_abcd_dataset_for_ast_w_possible_action_upperbound(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        context = []
        # state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}
        all_possible_actions = ['pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

        # get actions in this sample
        actions = []
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]
                actions.append(target_action)
        
        # add all actions to actions_for_context, randomly select 4 actions from all_possible_actions apart from actions to actions_for_context
        # actions_for_context = []
        # for action in actions:
        #     actions_for_context.append(action)
        # for i in range(4):
        #     action = random.choice(all_possible_actions)
        #     while action in actions_for_context:
        #         action = random.choice(all_possible_actions)
        #     actions_for_context.append(action)
        
        # # shuffle actions_for_context
        # random.shuffle(actions_for_context)
        counter_action = 0
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                actions_for_context = actions[counter_action:]
                counter_action += 1
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                # context.append(target_action + " " + str(target_values))
                # context.append("possible actions: " + '[' + ', '.join(actions_for_context) + ']')
                context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
            else:
                if turn["text"][-1] not in ['.', '?', '!']:
                    turn['text'] = turn['text'] + '.'
                context.append(turn["text"])
    return parsed_samples

'''
add possible action and sequence actions to context (upper bound)
'''
def parse_abcd_dataset_for_ast_w_possible_action_sequence_upperbound(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        # print("data type: ", data_type, "sample_no: ", sample_numbers)
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        context = []
        # state_mapping: {'s0': 'init', 's1': 'pull-up-account', 's2': 'enter-details', 's3': 'verify-identity', 's4': 'make-password', 's5': 'search-timing', 's6': 'search-policy', 's7': 'validate-purchase', 's8': 'search-faq', 's9': 'membership', 's10': 'search-boots', 's11': 'try-again', 's12': 'ask-the-oracle', 's13': 'update-order', 's14': 'promo-code', 's15': 'update-account', 's16': 'search-membership', 's17': 'make-purchase', 's18': 'offer-refund', 's19': 'notify-team', 's20': 'record-reason', 's21': 'search-jeans', 's22': 'shipping-status', 's23': 'search-shirt', 's24': 'instructions', 's25': 'search-jacket', 's26': 'log-out-in', 's27': 'select-faq', 's28': 'subscription-status', 's29': 'send-link', 's30': 'search-pricing', 's31': 'end'}
        all_possible_actions = ['pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

        # get actions in this sample
        actions = []
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]
                actions.append(target_action)
        
        # add all actions to actions_for_context, randomly select 4 actions from all_possible_actions apart from actions to actions_for_context
        # actions_for_context = []
        # for action in actions:
        #     actions_for_context.append(action)
        # for i in range(4):
        #     action = random.choice(all_possible_actions)
        #     while action in actions_for_context:
        #         action = random.choice(all_possible_actions)
        #     actions_for_context.append(action)
        
        # # shuffle actions_for_context
        # random.shuffle(actions_for_context)
        counter_action = 0
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                actions_for_context = actions[counter_action:]
                counter_action += 1
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                context.append(target_action + " " + str(target_values))
                # context.append("possible actions: " + '[' + ', '.join(actions_for_context) + ']')
                context_.append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
            else:
                if turn["text"][-1] not in ['.', '?', '!']:
                    turn['text'] = turn['text'] + '.'
                context.append(turn["text"])
    return parsed_samples


'''
add else
'''
def parse_abcd_dataset_for_ast(raw_data_path: Path, data: List, data_type: str, sample_numbers = 80):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []
    
    if data_type == "train":
        if sample_numbers.lower() == "all":
            sample_no = len(data)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)
        
    for sample in data:
        context = []
        # print("sample id: ", sample["convo_id"])
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                # print([target_action, target_values])
                target_action = sample['scenario']['flow'] + ":" + target_action
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"], "flow": sample["scenario"]["flow"]})
            else:
                context.append(turn["text"])
    return parsed_samples

def parse_abcd_dataset_for_ast_incremental(raw_data_path: Path, data: List, data_type: str):
    non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
    parsed_samples = []

    if data_type == "train":
        existing_conv_ids = []
        with open('/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/train_AST_abcd_10p.json', 'r') as file:
            for line in file:
                json_data = json.loads(line)
                if json_data['convo_id'] not in existing_conv_ids:
                    existing_conv_ids.append(json_data['convo_id'])
        print("existing_conv_ids: ", existing_conv_ids)

        sample_no = 4000
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        
        sample_ids_tmp = []
        for i in sample_ids:
            if data[i]['convo_id'] in existing_conv_ids:
                continue
            sample_ids_tmp.append(i)
        sample_ids = sample_ids_tmp

        if len(sample_ids) < sample_no:
            print("lack of samples, number of samples: ", sample_no - len(sample_ids))
            while len(sample_ids) < sample_no:
                sample_ids_tmp = random.sample(range(len(data)), 1)
                if data[sample_ids_tmp[0]]['convo_id'] in existing_conv_ids:
                    continue
                sample_ids.append(sample_ids_tmp[0])

        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)
        
    for sample in data:
        context = []
        # print("sample id: ", sample["convo_id"])
        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                potential_values = value_by_action[target_action]
                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    corrected_target_values = []
                    for value in target_values:
                        context_str = " ".join(context)
                        corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
                        corrected_target_values.append(corrected_value)
                    target_values = corrected_target_values
                context_=[c for c in context[:]]
                # print(context)
                # print([target_action, target_values])
                parsed_samples.append({"context": context_, "action": [target_action, target_values], "convo_id": sample["convo_id"], "turn_id": turn["turn_count"]})
            else:
                context.append(turn["text"])
    return parsed_samples


'''
by jianyuan
'''
# def parse_abcd_dataset_for_ast(raw_data_path: Path, data: List):
#     non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
#     parsed_samples = []
#     for sample in data:
#         context = []

#         for idx, turn in enumerate(sample["delexed"]):
#             if turn["speaker"] == "action":
#                 _, _, target_action, target_values, _ = turn["targets"]

#                 potential_values = value_by_action[target_action]
#                 if not target_values or target_values == [""]:
#                     target_values = ["none"]
#                 else:
#                     corrected_target_values = []
#                     for value in target_values:
#                         context_str = " ".join(context)
#                         corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
#                         corrected_target_values.append(corrected_value)
#                     target_values = corrected_target_values
#                 context_=[c for c in context[:-1]]
#                 parsed_samples.append({"context": context_, "action": [target_action, target_values]})
#             context.append(turn["text"])

#     return parsed_samples

'''
original implementation
'''
# def parse_abcd_dataset_for_ast(raw_data_path: Path, data: List):
#     non_enumerable, enumerable, mappers, value_by_action = prepare_labels_for_ast(raw_data_path)
#     parsed_samples = []
#     for sample in data:
#         context = []

#         for idx, turn in enumerate(sample["delexed"]):
#             if turn["speaker"] == "action":
#                 _, _, target_action, target_values, _ = turn["targets"]

#                 potential_values = value_by_action[target_action]
#                 if not target_values or target_values == [""]:
#                     target_values = ["none"]
#                 else:
#                     corrected_target_values = []
#                     for value in target_values:
#                         context_str = " ".join(context)
#                         corrected_value = ast_value_to_id(context_str, value, potential_values, enumerable)
#                         corrected_target_values.append(corrected_value)
#                     target_values = corrected_target_values

#                 parsed_samples.append({"context": context, "action": [target_action, target_values]})

#             context.append(turn["text"])

#     return parsed_samples

def get_value_mappings(data_path: Path):
    """
    Create the mapping from values like "shirt_how_1" to "remove a stain from the shirt" per agent guidelines
    Mapping validated from the original ABCD guidelines
    Reference: https://docs.google.com/document/d/1_SZit-iUAzNCICJ6qahULoMhqVOJCspQF37QiEJzHLc
    """

    def _get_value_mappings(subflows_data, value_prefixes, expected_value_count, get_value_type_fct=None):
        _value_mappings = {}
        for value in value_prefixes:
            added_values_count = 0

            # Skipping the first value since it holds an instruction to the agent not the value
            nl_values = subflows_data[f"{value} FAQ"]["instructions"][1].split(",")

            for idx, nl_value in enumerate(nl_values):
                value_type = get_value_type_fct(idx + 1) if get_value_type_fct else "_{idx + 1}"
                _value_mappings[f"{value.lower()}{value_type}"] = nl_value.strip()
                added_values_count += 1

            assert added_values_count == expected_value_count
        return _value_mappings

    guidelines = read_abcd_guidelines(data_path)

    value_mappings = {}

    single_item_queries_subflows = guidelines["Single-Item Query"]["subflows"]
    # There are 4 "how" values (e.g., shirt_how_4) followed by 4 "other" values (e.g., shirt_other_4)
    get_value_type = lambda i: f"_how_{i if i <= 4 else i - 4}" if i <= 4 else f"_other_{i if i <= 4 else i - 4}"
    value_mappings.update(
        _get_value_mappings(
            single_item_queries_subflows,
            ["Shirt", "Jacket", "Jeans", "Boots"],
            8,  # there 8 values for each
            get_value_type_fct=get_value_type,
        )
    )

    store_wide_queries_subflows = guidelines["Storewide Query"]["subflows"]
    value_mappings.update(
        _get_value_mappings(
            store_wide_queries_subflows,
            ["Policy", "Timing", "Pricing", "Membership"],
            4,  # there only value (i.e., timing_1, ..., timing_4)
            get_value_type_fct=get_value_type,
        )
    )

    return value_mappings


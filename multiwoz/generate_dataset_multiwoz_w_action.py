import sys
import argparse
import logging

import random
import re

import jsonlines
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from urllib.request import urlretrieve
from tqdm import tqdm as progress_bar
from datasets import load_dataset

import src.data.multiwoz_24_nlp_utils as multiwoz_24_nlp_utils
import src.data.abcd_utils as abcd_utils

# set seed 
random.seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def read_action_mapping_file(data_path: Path, prefix: str):
    action_mapping_file_path = data_path / f"{prefix}_action_mappings.json"
    with action_mapping_file_path.open() as f:
        action_mappings = json.load(f)
    return action_mappings

'''
original version
'''
def parse_abcd_dataset_for_workflow_discovery(data: List, action_mappings: Dict, value_mappings: Dict):
    parsed_samples = []
    for sample in data:
        original_dialogue = []
        workflow = []

        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    target_values = [value_mappings.get(v, v) for v in target_values if v.lower() != "n/a"]

                target_action = action_mappings[target_action]

                step_data = [target_action, target_values]

                if step_data not in workflow:  # Skipping annotation duplicates
                    workflow.append(step_data)

            else:
                # We use the original dialogue, since the delexed version has anonymized slot values (e.g., [username])
                original_dialogue.append(sample["original"][idx])

        parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

    return parsed_samples


'''
WD for getting the workflows for initialization
'''
def parse_abcd_dataset_for_workflow_discovery(data: List, action_mappings: Dict, value_mappings: Dict, data_type: str):
    parsed_samples = []

    if data_type == "train":
        sample_no = 80
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(data)), sample_no)
        print("data type: ", data_type, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_data = [data[i] for i in sample_ids]
        data = tmp_data
        # sample_no = len(data)
        # print("data type: ", data_type, "sample_no: ", sample_no)
    else:
        sample_no = len(data)
        print("data type: ", data_type, "sample_no: ", sample_no)

    for sample in data:
        original_dialogue = []
        workflow = []

        for idx, turn in enumerate(sample["delexed"]):
            if turn["speaker"] == "action":
                _, _, target_action, target_values, _ = turn["targets"]

                if not target_values or target_values == [""]:
                    target_values = ["none"]
                else:
                    target_values = [value_mappings.get(v, v) for v in target_values if v.lower() != "n/a"]

                target_action = action_mappings[target_action]

                step_data = [target_action, target_values]

                if step_data not in workflow:  # Skipping annotation duplicates
                    workflow.append(step_data)

            else:
                # We use the original dialogue, since the delexed version has anonymized slot values (e.g., [username])
                original_dialogue.append(sample["original"][idx])

        parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

    return parsed_samples

def convert_workflow_to_str(workflow: List):
    workflow_str_parts = []
    for action, values in workflow:
        workflow_str_parts.append(f"{action} [{', '.join(values)}]")

    workflow_str = "; ".join(workflow_str_parts)
    return workflow_str


def convert_dialogue_to_str(dialogue: List):
    dialogue_str_parts = []
    for speaker, utterance in dialogue:
        dialogue_str_parts.append(utterance)

    dialogue_str = "Dialogue: " + " ".join(dialogue_str_parts)
    return dialogue_str


def create_input_w_possible_actions(dialogue_str: str, all_actions: List):
    possible_actions = all_actions
    return dialogue_str + " Actions: " + ", ".join(possible_actions)


def create_input_w_possible_actions_plus(dialogue_str: str, all_actions: List, workflow: List, r_min: int = 10):
    random_actions = random.sample(all_actions, random.randint(r_min, len(all_actions)))
    actions = [a[0] for a in workflow]

    possible_actions_plus = list(set(random_actions + actions))
    random.shuffle(possible_actions_plus)

    return dialogue_str + " Actions: " + ", ".join(possible_actions_plus)


def create_workflow_discovery_split_dataset(parsed_data, all_actions):
    wd_split_data = []
    for idx, sample in enumerate(parsed_data):
        workflow = sample["workflow"]

        workflow_str = convert_workflow_to_str(workflow)
        dialogue_str = convert_dialogue_to_str(sample["original_dialogue"])
        input_w_possible_actions = create_input_w_possible_actions(dialogue_str, all_actions)
        input_w_possible_actions_plus = create_input_w_possible_actions_plus(dialogue_str, all_actions, workflow)

        wd_sample = {
            "sample_id": len(wd_split_data),
            "target": workflow_str,
            "input": dialogue_str,
            "input_w_possible_actions": input_w_possible_actions,  # i.e., w/ Possible Actions in paper
            "input_w_possible_actions_plus": input_w_possible_actions_plus,  # i.e., w/ Possible Actions+ in paper
            "target_data": json.dumps(workflow),  # Used during metrics evaluation
        }

        wd_split_data.append(wd_sample)

    return wd_split_data


def get_workflow_discovery_data_from_abcd(data_path: Path):
    raw_data = abcd_utils.read_abcd_raw_data(data_path)
    value_mappings = abcd_utils.get_value_mappings(data_path)
    action_mappings = read_action_mapping_file(data_path, "abcd")

    parsed_data = {}
    for split, split_data in raw_data.items():
        parsed_data[split] = parse_abcd_dataset_for_workflow_discovery(split_data, action_mappings, value_mappings, data_type=split)

    return parsed_data, list(action_mappings.values())


def write_split_data(processed_data_path: Path, dataset_name: str, split: str, data, task_name="workflow_discovery"):
    output_file = processed_data_path / f"{split}_{task_name}_{dataset_name}.json"
    with jsonlines.open(output_file.resolve(), mode="w") as writer:
        # We use jsonline to use the hugggingface dataset library
        for sample in data:
            writer.write(sample)


def create_workflow_discovery_dataset(
    processed_data_path: Path, parsed_data: Dict, all_actions: List, dataset_name: str
):
    for split, split_data in parsed_data.items():
        wd_split_data = create_workflow_discovery_split_dataset(split_data, all_actions)

        write_split_data(processed_data_path, dataset_name, split, wd_split_data)


def read_multiwoz_raw_data(raw_data_path: Path, split: str, sample_numbers: str):
    file_path = raw_data_path / f"{split}_multiwoz_22.json"
    with jsonlines.open(file_path) as reader:
        samples = [s for s in reader]
    
    if split == "train":
        if sample_numbers.lower() == "all":
            sample_no = len(samples)
        else:
            sample_no = int(sample_numbers)
        # randomly sample 80 data for training
        sample_ids = random.sample(range(len(samples)), sample_no)
        print("data type: ", split, "sample_no: ", sample_no, "sample_ids: ", sample_ids)
        tmp_samples = [samples[i] for i in sample_ids]
        samples = tmp_samples
        # sample_no = len(samples)
        # print("data type: ", split, "sample_no: ", sample_no)
        # print(samples)
    else:
        sample_no = len(samples)
        print("data type: ", split, "sample_no: ", sample_no)

    return samples


def is_empty_value(value_list):
    """Following MutliWOZ 2.4"""
    return any(
        [
            value in ["dont care", "dontcare", "don't care", "do not care", "not_mentioned", "none", "unknown"]
            for value in value_list
        ]
    )


def get_frame_slot_values(frame, replacements):
    parsed_slot_values = {}
    for slot_name, slot_value in zip(
        frame["slots_values"]["slots_values_name"], frame["slots_values"]["slots_values_list"]
    ):
        if is_empty_value(slot_value):
            continue
        parsed_slot_values[slot_name] = [multiwoz_24_nlp_utils.normalize(s, replacements) for s in slot_value]
    return parsed_slot_values


def get_slot_value_name(multiwoz_name, domain, act):
    match = re.match(f"^{domain}-{act}(.*)$", multiwoz_name)
    if not match:
        match = re.match(f"^{domain}-(.*)$", multiwoz_name)
        if not match:
            raise ValueError()

    return match.group(1)


def find_best_value(name, domain, act, values, dialogue_utterances_str):
    best_value = None
    if len(values) == 0:
        raise ValueError()

    for value in values:
        try:
            value = int(value)
            # Integer value

            exact_name = get_slot_value_name(name, domain, act)
            best_value = str(value) + " " + exact_name  # e.g., replace 2 to 2 people
            break
        except:
            pass

        if value in ["yes", "no"]:
            exact_name = get_slot_value_name(name, domain, act)
            # e.g., replace internet yes -> with internet
            best_value = "with" if value == "yes" else "without"
            best_value += " "
            best_value += exact_name
            break

        if value in dialogue_utterances_str:
            # In MultiWoz 2.2 some values have multiple candidate, we choose the one that exists in the dialogue
            best_value = value
            break

    if best_value is None:
        # Probably an annotation error
        best_value = values[0]

    return best_value


def convert_intents_to_workflow(dialogue_intents_w_values, original_dialogue: List, action_mappings: Dict):
    dialogue_str = " ".join([u[1] for u in original_dialogue])
    workflow = []
    intents = list(dialogue_intents_w_values.keys())
    for intent in intents:
        act, domain = intent.split("_")
        slot_values = dialogue_intents_w_values[intent]

        action_name = action_mappings[intent]
        # print(f"intent: {intent}, act: {act}, domain: {domain}, action_name: {action_name}")
        action_values = []
        for name, values in slot_values.items():
            # Following Mutliwoz 2.4 where the "book" slot values are linked to the book intent
            if act == "book":
                match = re.match(f"^{domain}-book(.*)$", name)
                if not match:
                    continue
                value = find_best_value(name, domain, act, values, dialogue_str)
                action_values.append(value)
            else:
                if "book" in name:
                    continue
                value = find_best_value(name, domain, act, values, dialogue_str)
                action_values.append(value)

        workflow.append([action_name, action_values])
        # use intent: xxx_xxxx
        # instead of: xxx xx xxxxxx
        # workflow.append([intent, action_values])

    return workflow

def convert_intents_to_ast(dialogue_intents_w_values, original_dialogue: List, action_mappings: Dict, turn_intent: Dict, chain_action_candidates: Dict, sample_states: Dict):

    action_mapping_reverse = {v: k for k, v in action_mappings.items()}
    # from aalpy.utils import load_automaton_from_file
    # import numpy as np

    # # load an automaton
    # automaton = load_automaton_from_file("./chainPrior/learned_mdp_multiwoz_all.dot", automaton_type='mdp')
    # # print(automaton)
    # # visualize the automaton
    # # visualize_automaton(automaton)
    # automaton = str(automaton)
    # # print(automaton)

    # automaton_splits = automaton.split('\n')
    # # print(automaton_splits)
    # automaton_states = automaton_splits[1:15]
    # # ['s0 [label="init"];', 's1 [label="pull-up-account"];']
    # automaton_transitions = automaton_splits[15:-4]
    # # ['s0 -> s0  [label="init:1.0"];', 's0 -> s1  [label="action:0.03"];']

    # state_mapping = {}
    # for state in automaton_states:
    #     state_name = state.split(' ')[0]
    #     state_label = state.split('[label="')[1].split('"];')[0]
    #     state_mapping[state_name] = state_label

    # all_possible_actions = ['init','find_hotel', 'book_hotel', 'find_train', 'book_train', 'find_attraction', 'find_restaurant', 'book_restaurant', 'find_hospital', 'book_taxi', 'find_taxi', 'find_bus', 'find_police']

    # transition_mapping = {}
    # chain_action_candidates = {}
    # for transition in automaton_transitions:
    #     transition_split = transition.split('->')
    #     source_state = transition_split[0].strip()
    #     target_state = transition_split[1].strip().split(' ')[0]
    #     transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
    #     transition_action = transition_label.split(':')[0]
    #     transition_freq = float(transition_label.split(':')[1])
    #     transition_mapping[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)
    #     if state_mapping[source_state] in all_possible_actions:
    #         if state_mapping[source_state] not in chain_action_candidates:
    #             chain_action_candidates[state_mapping[source_state]] = []
    #         if state_mapping[target_state] in all_possible_actions and transition_freq > 0.0:
    #             chain_action_candidates[state_mapping[source_state]].append((state_mapping[target_state], transition_freq))
    #             # chain_action_candidates[state_mapping[source_state]].append(state_mapping[target_state])
    #         else:
    #             pass
    #     else:
    #         continue
    
    # for paction in all_possible_actions:
    #     if paction not in chain_action_candidates:
    #         chain_action_candidates[paction] = []
    #     if chain_action_candidates[paction] == []:
    #         chain_action_candidates[paction] = all_possible_actions

    # # order the actions by the frequency of the transition, and select the most possible actions, whose sum of all the selected frequency is greater than 0.8
    # for key, value in chain_action_candidates.items():
    #     # print(f"key: {key}, value: {value}")
    #     value = value[1:]
    #     if len(value) == 0:
    #         continue
    #     if not isinstance(value[0], tuple):
    #         continue
    #     value.sort(key=lambda x: x[1], reverse=True)
    #     selected_actions = []
    #     freq_sum = 0.0
    #     for action, freq in value:
    #         if freq_sum + freq <= 0.9:
    #             selected_actions.append([action, freq])
    #             freq_sum += freq
    #         elif freq_sum + freq > 0.9 and freq_sum < 0.9:
    #             selected_actions.append([action, freq])
    #             freq_sum += freq
    #         else:
    #             break
    #     chain_action_candidates[key] = selected_actions

    # from s0 to s31, if some pair of states are not in the transition_mapping, then the frequency is 0
    # for i in range(32):
    #     for j in range(32):
    #         if (state_mapping[f's{i}'], state_mapping[f's{j}']) not in transition_mapping:
    #             transition_mapping[(state_mapping[f's{i}'], state_mapping[f's{j}'])] = ('unknown', 0.0)

    # print(f"transition_mapping: {transition_mapping}")
    # print(f"chain_action_candidates: {chain_action_candidates}")
    # for key, value in chain_action_candidates.items():
    #     print(f"key: {key}, value: {value}")


    # print("turn_intent: ", turn_intent)
    # print(original_dialogue)
    # print()
    dialogue_str = " ".join([u[1] for u in original_dialogue])
    ASTs = []
    intents = list(dialogue_intents_w_values.keys())
    # turn_intent_values = list(turn_intent.values())
    # turn_intent_keys = list(turn_intent.keys())
    turn_intent_values = []
    turn_intent_keys = []
    for key, value in turn_intent.items():
        turn_intent_values.extend(value)
        turn_intent_keys.extend([key] * len(value))
    
    last_action = "init"
    for i in range(len(intents)):
        intent = intents[i]
        turn_state = sample_states[intent]
        # find the index of the last position that has the same intent in the turn_intent
        index_turn_intent_values = [i for i, x in enumerate(turn_intent_values) if x == intent]
        # print("turn_intent_values: ", turn_intent_values)
        # print("intent: ", intent)
        # print("index_turn_intent_values: ", index_turn_intent_values)
        # print("intent: ", intent)
        # print("turn_intent_keys: ", turn_intent_keys)
        # print("index_turn_intent_values: ", index_turn_intent_values)
        # print("turn_intent_values: ", turn_intent_values)
        # print()

        last_turn_id = int(turn_intent_keys[index_turn_intent_values[-1]])
        # print("last_turn_id: ", last_turn_id)
        act, domain = intent.split("_")
        slot_values = dialogue_intents_w_values[intent]
        # print(f"intent: {intent}, act: {act}, domain: {domain}, slot_values: {slot_values}")

        action_name = action_mappings[intent]
        # print(f"intent: {intent}, act: {act}, domain: {domain}, action_name: {action_name}")
        action_values = []
        for name, values in slot_values.items():
            # Following Mutliwoz 2.4 where the "book" slot values are linked to the book intent
            if act == "book":
                match = re.match(f"^{domain}-book(.*)$", name)
                if not match:
                    continue
                value = find_best_value(name, domain, act, values, dialogue_str)
                action_values.append(value)
            else:
                if "book" in name:
                    continue
                value = find_best_value(name, domain, act, values, dialogue_str)
                action_values.append(value)
        
        # for testing
        tmp_ast_sample = {}
        tmp_ast_sample["action"] = [turn_state+":"+action_name, action_values]
        tmp_ast_sample["flow"] = turn_state
        # print("last_turn_id: ", last_turn_id)
        # print("ast dialogue: ", original_dialogue[:last_turn_id+1])
        tmp_ast_sample["context"] = [u[1] for u in original_dialogue[:last_turn_id+1]]
        original_dialogue[last_turn_id][1] = original_dialogue[last_turn_id][1] + " " + action_name + " [" + ", ".join(action_values) + "]"

        # for action in chain_action_candidates[last_action]:
        #     print(f"action: {action[0]}, freq: {action[1]}")
        actions_for_context = [f"{action[1]:.2f}*{action_mappings[action[0]]}" for action in chain_action_candidates[last_action]]
        last_action = intent

        # tmp_ast_sample["context"].append("Possible Actions: " + '[' + ', '.join(actions_for_context) + ']')
        tmp_ast_sample["turn_id"] = last_turn_id
        # use intent: xxx_xxxx
        # instead of: xxx xx xxxxxx
        ASTs.append([tmp_ast_sample['turn_id'], tmp_ast_sample['action'], tmp_ast_sample['context'], tmp_ast_sample['flow']])

    return ASTs


def parse_multiwoz_dataset(raw_data: List, action_mappings: Dict, replacements: List):
    parsed_samples = []

    for sample in raw_data:
        original_dialogue = []
        intents_w_slots = defaultdict(dict)  # Using python3.8, dictionaries keep insert order
        turns = sample["turns"]
        for speaker, utterance, active_frames in zip(turns["speaker"], turns["utterance"], turns["frames"]):
            utterance = multiwoz_24_nlp_utils.normalize(utterance, replacements)

            original_dialogue.append(["user" if speaker == 0 else "system", utterance])

            if speaker == 0:  # User
                for frame in active_frames["state"]:
                    active_intent = frame["active_intent"]
                    if active_intent.lower() == "none":
                        continue

                    parsed_slot_values = get_frame_slot_values(frame, replacements)
                    intents_w_slots[active_intent].update(parsed_slot_values)

        workflow = convert_intents_to_workflow(intents_w_slots, original_dialogue, action_mappings)

        parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

    return parsed_samples


'''
Use the actions prepared for the workflow discovory in the workflow discovery work
It may contains less action states than the actions in the original dialogue turns, cause it merges some actions into one
'''
def parse_multiwoz_dataset_for_ast(raw_data: List, action_mappings: Dict, replacements: List):
    parsed_samples = []

    from aalpy.utils import load_automaton_from_file
    import numpy as np

    # load an automaton
    automaton = load_automaton_from_file("./chainPrior/learned_mdp_multiwoz_all.dot", automaton_type='mdp')
    # print(automaton)
    # visualize the automaton
    # visualize_automaton(automaton)
    automaton = str(automaton)
    # print(automaton)

    automaton_splits = automaton.split('\n')
    # print(automaton_splits)
    automaton_states = automaton_splits[1:15]
    # ['s0 [label="init"];', 's1 [label="pull-up-account"];']
    automaton_transitions = automaton_splits[15:-4]
    # ['s0 -> s0  [label="init:1.0"];', 's0 -> s1  [label="action:0.03"];']

    state_mapping = {}
    for state in automaton_states:
        state_name = state.split(' ')[0]
        state_label = state.split('[label="')[1].split('"];')[0]
        state_mapping[state_name] = state_label

    all_possible_actions = ['init','find_hotel', 'book_hotel', 'find_train', 'book_train', 'find_attraction', 'find_restaurant', 'book_restaurant', 'find_hospital', 'book_taxi', 'find_taxi', 'find_bus', 'find_police']

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

    # order the actions by the frequency of the transition, and select the most possible actions, whose sum of all the selected frequency is greater than 0.8
    for key, value in chain_action_candidates.items():
        # print(f"key: {key}, value: {value}")
        value = value[1:]
        if len(value) == 0:
            continue
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
        chain_action_candidates[key] = selected_actions

    for sample in raw_data:
        # print("sample: ", json.dumps(sample, indent=2))
        original_dialogue = []
        intents_w_slots = defaultdict(dict)  # Using python3.8, dictionaries keep insert order
        turns = sample["turns"]
        turn_intent = {}
        current_state = []
        sample_states = {}
        for turn_id, speaker, utterance, active_frames in zip(turns["turn_id"], turns["speaker"], turns["utterance"], turns["frames"]):
            utterance = multiwoz_24_nlp_utils.normalize(utterance, replacements)

            original_dialogue.append(["user" if speaker == 0 else "system", utterance])

            if speaker == 0:  # User
                user_service = active_frames["service"]
                for k in range(len(user_service)):
                    if user_service[k] not in current_state:
                        current_state.append(user_service[k])
                for frame in active_frames["state"]:
                    active_intent = frame["active_intent"]
                    # print("active_intent: ", active_intent)
                    if active_intent.lower() == "none":
                        turn_intent[turn_id] = active_intent.lower()
                        continue

                    parsed_slot_values = get_frame_slot_values(frame, replacements)
                    intents_w_slots[active_intent].update(parsed_slot_values)

                    
                    if turn_id not in turn_intent:
                        turn_intent[turn_id] = []
                    if active_intent not in turn_intent[turn_id]:
                        turn_intent[turn_id].append(active_intent)

                    # if active_intent != last_turn_intent:
                    current_state.sort()
                    state_name = '-'.join(current_state)
                    sample_states[active_intent] = state_name
                    # last_turn_intent = active_intent

        print(f"len(sample_states): {len(sample_states.keys())}, len(intents_w_slots): {len(intents_w_slots)}, len(turn_intent): {len(turn_intent)}")
        ASTs = convert_intents_to_ast(intents_w_slots, original_dialogue, action_mappings, turn_intent, chain_action_candidates, sample_states)

        # parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

        for AST in ASTs:
            parsed_samples.append({"context": AST[2], "action": AST[1], "convo_id": sample["dialogue_id"], "turn_id": AST[0], "flow": AST[3]})


    return parsed_samples

'''
Use the actions that for each turn of users
'''
def parse_multiwoz_dataset_for_ast_turnBase(raw_data: List, action_mappings: Dict, replacements: List):
    parsed_samples = []

    for sample in raw_data:
        # print("sample: ", json.dumps(sample, indent=2))
        original_dialogue = []
        intents_w_slots = defaultdict(dict)  # Using python3.8, dictionaries keep insert order
        turns = sample["turns"]
        turn_intent = {}
        for turn_id, speaker, utterance, active_frames in zip(turns["turn_id"], turns["speaker"], turns["utterance"], turns["frames"]):
            utterance = multiwoz_24_nlp_utils.normalize(utterance, replacements)

            original_dialogue.append(["user" if speaker == 0 else "system", utterance])

            if speaker == 0:  # User
                for frame in active_frames["state"]:
                    active_intent = frame["active_intent"]
                    # print("active_intent: ", active_intent)
                    if active_intent.lower() == "none":
                        turn_intent[turn_id] = active_intent.lower()
                        continue

                    parsed_slot_values = get_frame_slot_values(frame, replacements)

                    act, domain = active_intent.split("_")
                    slot_values = parsed_slot_values

                    # print(f"intent: {intent}, act: {act}, domain: {domain}, action_name: {action_name}")
                    action_values = []
                    dialogue_str = " ".join([u[1] for u in original_dialogue])
                    for name, values in slot_values.items():
                        # Following Mutliwoz 2.4 where the "book" slot values are linked to the book intent
                        if act == "book":
                            match = re.match(f"^{domain}-book(.*)$", name)
                            if not match:
                                continue
                            value = find_best_value(name, domain, act, values, dialogue_str)
                            action_values.append(value)
                        else:
                            if "book" in name:
                                continue
                            value = find_best_value(name, domain, act, values, dialogue_str)
                            action_values.append(value)
                    
                    # parsed_slot_values:  {'hotel-parking': ['yes'], 'hotel-pricerange': ['cheap']}
                    # print("parsed_slot_values: ", parsed_slot_values)
                    # intents_w_slots[active_intent].update(parsed_slot_values)

                    
                    # if turn_id not in turn_intent:
                    #     turn_intent[turn_id] = []
                    # if active_intent not in turn_intent[turn_id]:
                    #     turn_intent[turn_id].append(active_intent)

                    tmp_sample = {}
                    tmp_sample["convo_id"] = sample["dialogue_id"]
                    tmp_sample["turn_id"] = turn_id
                    tmp_sample["action"] = [active_intent, action_values]
                    tmp_sample["context"] = [u[1] for u in original_dialogue[:int(turn_id)+1]]
                    tmp_sample["context"].append("Possible Actions: []")
                    
                    if len(parsed_samples) != 0 and tmp_sample["action"] == parsed_samples[-1]["action"]:
                        continue
                    parsed_samples.append(tmp_sample)

        # ASTs = convert_intents_to_ast(intents_w_slots, original_dialogue, action_mappings, turn_intent)

        # parsed_samples.append({"original_dialogue": original_dialogue, "workflow": workflow})

        # for AST in ASTs:
            # parsed_samples.append({"context": AST[2], "action": AST[1], "convo_id": sample["dialogue_id"], "turn_id": AST[0]})


    return parsed_samples


def get_workflow_discovery_data_from_multiwoz(raw_data_path: Path, dataset_name: str):
    action_mappings = read_action_mapping_file(raw_data_path, dataset_name)
    replacements = multiwoz_24_nlp_utils.get_replacements(raw_data_path)

    parsed_data = {}
    for split in ["train", "validation", "test"]:
        split_raw_data = read_multiwoz_raw_data(raw_data_path, split)
        parsed_data[split] = parse_multiwoz_dataset(split_raw_data, action_mappings, replacements)

    return parsed_data, list(action_mappings.values())


def get_ast_data_from_multiwoz(raw_data_path: Path, dataset_name: str, sample_numbers = 80):
    action_mappings = read_action_mapping_file(raw_data_path, dataset_name)
    replacements = multiwoz_24_nlp_utils.get_replacements(raw_data_path)

    parsed_data = {}
    '''
    sample 1:  {'context': ['hello, how may i help you?', 'i want to know the state of my refund.', 'let me help you with that.', 'i have an existing refund of $100 + i want to refund another $<amount>.', 'did you want to add an extra item to your current refund?', 'yes.', 'could i have your full name or account id?', 'albert sanders.', 'account id 123445.', 'Possible Actions: []'], 'action': ['pull-up-account', ['albert sanders']], 'convo_id': 1746, 'turn_id': 10}
    sample 2:  {'context': ['hello, how may i help you?', 'i want to know the state of my refund.', 'let me help you with that.', 'i have an existing refund of $100 + i want to refund another $<amount>.', 'did you want to add an extra item to your current refund?', 'yes.', 'could i have your full name or account id?', 'albert sanders.', 'account id 123445.', "pull-up-account ['albert sanders']", 'thanks.', 'could i have your username, email address and order id to validate your order?', '<username>.', '<email>.', 'and the order id?', '<order_id>.', 'Possible Actions: [verify-identity, validate-purchase, record-reason, enter-details]'], 'action': ['validate-purchase', ['<username>', '<username>', '<username>']], 'convo_id': 1746, 'turn_id': 17}
    '''
    for split in ["train", "validation", "test"]:
        split_raw_data = read_multiwoz_raw_data(raw_data_path, split, sample_numbers)
        parsed_data[split] = parse_multiwoz_dataset_for_ast(split_raw_data, action_mappings, replacements)

        # if split == 'train':
        #     # print()
        #     # print("train data: ", parsed_data[split])
        #     # print()
        #     break

    return parsed_data
    


def get_ast_data_from_abcd(raw_data_path: Path, sample_numbers = 80):
    raw_data = abcd_utils.read_abcd_raw_data(raw_data_path)
    parsed_data = {}
    for split, split_data in raw_data.items():
        # train / dev / test
        # print(split)
        # parsed_data[split] = abcd_utils.parse_abcd_dataset_for_ast_w_action(raw_data_path, split_data, data_type=split, sample_numbers=sample_numbers)
        parsed_data[split] = abcd_utils.parse_abcd_dataset_for_ast_w_mostpossible_chain_action_aug(raw_data_path, split_data, data_type=split, sample_numbers=sample_numbers)
        # if split == "train":
        #     print("sample 1: ", parsed_data[split][0])
        #     print("sample 2: ", parsed_data[split][1])
        #     print()
    return parsed_data


def get_cds_data_from_abcd(raw_data_path: Path):
    raw_data = abcd_utils.read_abcd_raw_data(raw_data_path)
    parsed_data = {}
    for split, split_data in raw_data.items():
        parsed_data[split] = abcd_utils.parse_abcd_dataset_for_cds(raw_data_path, split_data)

    return parsed_data


def convert_context_to_str(context):
    return "Context: " + " ".join(context)


def convert_cds_context_to_str(context, utterance_candidates):
    context_str = convert_context_to_str(context)
    if utterance_candidates:
        context_str += " Candidates: "
        context_str += " ".join(utterance_candidates)

    return context_str


def create_cds_split_dataset(parsed_data: List):
    cds_split_data = []
    for idx, sample in enumerate(parsed_data):
        context = sample["context"]
        candidates = sample["candidates"]
        next_step = sample["next_step"]

        context_str = convert_cds_context_to_str(context, candidates)

        target_parts = [sample["intent"], next_step]
        if next_step == "respond":
            target_parts.append(sample["target_utterance"])
        elif next_step == "action":
            target_parts.append(convert_workflow_to_str([sample["take_action_target"]]))

        target_str = "; ".join(target_parts)

        cds_sample = {
            "sample_id": len(cds_split_data),
            "convo_id": sample["convo_id"],
            "turn_id": sample["turn_id"],
            "target": target_str,
            "input": context_str,
            "target_data": json.dumps(sample),  # Used during metrics evaluation
        }

        cds_split_data.append(cds_sample)
    return cds_split_data


def create_ast_split_dataset(parsed_data: List):
    ast_split_data = []
    for idx, sample in enumerate(parsed_data):
        action = sample["action"]
        context = sample["context"]

        action_str = convert_workflow_to_str([action])  # Same format as worklow with a single action
        context_str = convert_context_to_str(context)

        ast_sample = {
            "sample_id": len(ast_split_data),
            "convo_id": sample["convo_id"],
            "turn_id": sample["turn_id"],
            "flow": sample["flow"],
            "target": action_str,
            "input": context_str,
            "target_data": json.dumps(action),  # Used during metrics evaluation
        }

        ast_split_data.append(ast_sample)
    return ast_split_data


def create_ast_dataset(processed_data_path: Path, parsed_ast_data: Dict, dataset_name: str):
    for split, split_data in parsed_ast_data.items():
        wd_split_data = create_ast_split_dataset(split_data)
        write_split_data(processed_data_path, dataset_name, split, wd_split_data, task_name="AST")


def create_cds_dataset(processed_data_path: Path, parsed_cds_data: Dict, dataset_name: str):
    for split, split_data in parsed_cds_data.items():
        wd_split_data = create_cds_split_dataset(split_data)
        write_split_data(processed_data_path, dataset_name, split, wd_split_data, task_name="CDS")


def download_multiwoz_22_raw_data(raw_data_path: Path):
    dataset = load_dataset("multi_woz_v22")
    for split, data in dataset.items():
        file_path = raw_data_path / f"{split}_multiwoz_22.json"
        data.to_json(file_path)

    # Download replacement file
    replacement_file_path = raw_data_path / "mapping.pair"
    urlretrieve("https://raw.githubusercontent.com/smartyfh/MultiWOZ2.4/main/utils/mapping.pair", replacement_file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--raw_data_folder", required=True, help="Raw datasets folder path")
    parser.add_argument("-o", "--processed_data_folder", required=True, help="Processed datasets folfer path (output)")
    parser.add_argument("-n", "--sample_numbers", required=True, help="sample numbers for training data")
    return parser.parse_args()


def create_all_datasets(raw_data_path: Path, processed_data_path: Path, sample_numbers = 80):
    print("Creating datasets, this takes a while...")

    # print("Creating datasets from ABCD ...")
    # wd_data, all_actions = get_workflow_discovery_data_from_abcd(raw_data_path)
    # create_workflow_discovery_dataset(processed_data_path, wd_data, all_actions, "abcd")

    # ast_data = get_ast_data_from_abcd(raw_data_path, sample_numbers)
    # create_ast_dataset(processed_data_path, ast_data, "abcd")

    # cds_data = get_cds_data_from_abcd(raw_data_path)
    # create_cds_dataset(processed_data_path, cds_data, "abcd")

    # print("Creating datasets from MultiWOZ ...")
    # download_multiwoz_22_raw_data(raw_data_path)
    # wd_data, all_actions = get_workflow_discovery_data_from_multiwoz(raw_data_path, "multiwoz")
    # create_workflow_discovery_dataset(processed_data_path, wd_data, all_actions, "multiwoz")

    print("Creating datasets from Multi-woz ...")
    ast_data = get_ast_data_from_multiwoz(raw_data_path, "multiwoz", sample_numbers=sample_numbers)
    create_ast_dataset(processed_data_path, ast_data, "multiwoz")

    print("Done! Happy discovery")


if __name__ == "__main__":
    args = parse_args()
    create_all_datasets(Path(args.raw_data_folder), Path(args.processed_data_folder), args.sample_numbers)

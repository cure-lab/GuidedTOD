import os
import json
from aalpy.utils import load_automaton_from_file, save_automaton_to_file, visualize_automaton, generate_random_dfa
from aalpy.automata import Dfa
from aalpy.SULs import AutomatonSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_Lstar, run_KV
import time

start = time.time()
# load json file and read line by line
def load_json_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# convert transition_matrix to dot format
def convert_to_dot(transition_matrix):
    dot_str = "digraph learned_mdp {\n"
    dot_str += "s0 [label=\"init\"];\n"
    for i in range(len(possible_states)):
        dot_str += "s" + str(i+1) + " [label=\"" + possible_states[i] + "\"];\n"
    dot_str += "s45 [label=\"end\"];\n"    

    dot_str += "s0 -> s0  [label=\"init:1.0\"];\n"
    # dot_str += "s0 -> s1  [label=\"action:0.215\"];\n"
    # dot_str += "s0 -> s2  [label=\"action:0.00125\"];\n"
    # dot_str += "s0 -> s3  [label=\"action:0.22875\"];\n"
    # dot_str += "s0 -> s4  [label=\"action:0.0\"];\n"
    # dot_str += "s0 -> s5  [label=\"action:0.1675\"];\n"
    # dot_str += "s0 -> s6  [label=\"action:0.2875\"];\n"
    # dot_str += "s0 -> s7  [label=\"action:0.00125\"];\n"
    # dot_str += "s0 -> s8  [label=\"action:0.03625\"];\n"
    # dot_str += "s0 -> s9  [label=\"action:0.0\"];\n"
    # dot_str += "s0 -> s10  [label=\"action:0.0325\"];\n"
    # dot_str += "s0 -> s11  [label=\"action:0.0\"];\n"
    # dot_str += "s0 -> s12  [label=\"action:0.03\"];\n"
    for i in range(len(possible_states)):
        if possible_states[i] in action_rate:
            dot_str += "s0 -> s" + str(i+1) + "  [label=\"action:" + str(action_rate[possible_states[i]]) + "\"];\n"
        else:
            dot_str += "s0 -> s" + str(i+1) + "  [label=\"action:0.0\"];\n"
    dot_str += "s0 -> s45  [label=\"end:1\"];\n"

    for j in range(len(possible_states)):
        index = j
        state1 = possible_states[j]
        dot_str += "s" + str(index+1) + " -> s0  [label=\"init:1.0\"];\n"
        for k in range(len(possible_states)):
            index2 = k
            state2 = possible_states[k]
            if transition_matrix[state1][state2] != 0:
                dot_str += "s" + str(index+1) + " -> s" + str(index2+1) + "  [label=\"" + "action" + ":" + str(transition_matrix[state1][state2]) + "\"];\n"
        dot_str += 's' + str(index+1) + ' -> s45' + ' [label="end:1.0"];\n'
    
    dot_str += 's45 -> s45 [label="init:1.0"];\n'
    dot_str += 's45 -> s45  [label="action:1.0"];\n'
    dot_str += 's45 -> s45  [label="end:1.0"];\n'

    dot_str += "__start0 [label=\"\", shape=none];\n"
    dot_str += "__start0 -> s0  [label=\"\"];\n"
    dot_str += "}"

    return dot_str

data_part = "10p"
train_path = f"../../data/processed/train_AST_abcd_waction_flow_{data_part}.json"

train_data = load_json_file(train_path)

mapping_dict = {
  "pull-up-account": "pull up the costumer account",
  "enter-details": "enter details",
  "verify-identity": "verify costumer identity",
  "make-password": "create new password",
  "search-timing": "search timing",
  "search-policy": "check policy",
  "validate-purchase": "validate purchase",
  "search-faq": "search faq",
  "membership": "check membership level",
  "search-boots": "search for boots",
  "try-again": "ask costumer to try again",
  "ask-the-oracle": "ask oracle",
  "update-order": "update order information",
  "promo-code": "offer promo code",
  "update-account": "update costumer account",
  "search-membership": "get memberships information",
  "make-purchase": "make purchase",
  "offer-refund": "offer refund",
  "notify-team": "notify team",
  "record-reason": "record reason",
  "search-jeans": "search for jeans",
  "shipping-status": "get shipping status",
  "search-shirt": "search for shirt",
  "instructions": "check instructions",
  "search-jacket": "search for jacket",
  "log-out-in": "ask costumer log out log in",
  "select-faq": "select topic in faq",
  "subscription-status": "get subscription status",
  "send-link": "send link to costumer",
  "search-pricing": "check pricing"
}

# conver key to value and value to key
def convert_dict(mapping_dict):
    new_dict = {}
    for key, value in mapping_dict.items():
        new_dict[value] = key
    return new_dict

new_mapping_dict = convert_dict(mapping_dict)

# load a json file not line by line
guideline_path = "../../data/raw/guidelines.json"
with open(guideline_path, 'r') as f:
    guidelines = json.load(f)

valid_actions = []
for scenario in guidelines.keys():
    scenario_flows = guidelines[scenario]
    subflows = scenario_flows['subflows']
    # print(subflows)
    subflows_names = subflows.keys()
    for subflow_name in subflows_names:
        subflow = subflows[subflow_name]
        actions = subflow['actions']
        # valid_actions.extend(actions)
        for action in actions:
            action_name = action['button']
            action_name = action_name.lower()
            action_name = action_name.replace(" ", "-")
            if action_name not in valid_actions:
                valid_actions.append(action_name)
                print(action_name)

all_possible_actions = ["pull-up-account", "enter-details", "verify-identity", "make-password", "search-timing", "search-policy", "validate-purchase", "search-faq", "membership", "search-boots", "try-again", "ask-the-oracle", "update-order", "promo-code", "update-account", "search-membership", "make-purchase", "offer-refund", "notify-team", "record-reason", "search-jeans", "shipping-status", "search-shirt", "instructions", "search-jacket", "log-out-in", "select-faq", "subscription-status", "send-link", "search-pricing"]

# merge two lists
all_possible_actions = list(set(valid_actions + all_possible_actions))

all_transitions = []
tmp_all_transitions = {}

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

for flow_name in all_flow_name:
    # flow_name = "storewide_query"

    for i in range(len(train_data)):
        if train_data[i]['flow'] == flow_name:
            tmp_train_data = train_data[i]['target'].split(':')[1].strip()
            action = tmp_train_data.split(' ')[0].strip()
            if train_data[i]['convo_id'] not in tmp_all_transitions:
                tmp_all_transitions[train_data[i]['convo_id']] = []
            tmp_all_transitions[train_data[i]['convo_id']].append(action)

    for key, value in tmp_all_transitions.items():
        all_transitions.append(value)

    # count the freequency of the first action in each dialogue
    action_count = {}
    for i in range(len(all_transitions)):
        if all_transitions[i][0] not in action_count:
            action_count[all_transitions[i][0]] = 1
        else:
            action_count[all_transitions[i][0]] += 1

    # convert to rate
    action_rate = {}
    total_dialogue = len(all_transitions)
    for key, value in action_count.items():
        action_rate[key] = value / total_dialogue

    possible_states = all_possible_actions
    possible_states = list(possible_states)
    # ['pull-up-account', 'enter-details', 'verify-identity', 'make-password', 'search-timing', 'search-policy', 'validate-purchase', 'search-faq', 'membership', 'search-boots', 'try-again', 'ask-the-oracle', 'update-order', 'promo-code', 'update-account', 'search-membership', 'make-purchase', 'offer-refund', 'notify-team', 'record-reason', 'search-jeans', 'shipping-status', 'search-shirt', 'instructions', 'search-jacket', 'log-out-in', 'select-faq', 'subscription-status', 'send-link', 'search-pricing']

    # count the transition frequency between every two actions/state according to all_transitions
    transition_count = {}
    for i in range(len(all_transitions)):
        for j in range(len(all_transitions[i]) - 1):
            if (all_transitions[i][j], all_transitions[i][j+1]) in transition_count:
                transition_count[(all_transitions[i][j], all_transitions[i][j+1])] += 1
            else:
                transition_count[(all_transitions[i][j], all_transitions[i][j+1])] = 1

    # if no transition between two states, set the frequency to 0
    for state1 in possible_states:
        for state2 in possible_states:
            if (state1, state2) not in transition_count:
                transition_count[(state1, state2)] = 0

    # count the frequency of each state
    transition_matrix = {}
    for state1 in possible_states:
        transition_matrix[state1] = {}
        for state2 in possible_states:
            transition_matrix[state1][state2] = 0
        
        for key, value in transition_count.items():
            if state1 == key[0]:
                transition_matrix[state1][key[1]] = value

    # convert count to probability
    for state1 in possible_states:
        sum_count = sum(transition_matrix[state1].values())
        for state2 in possible_states:
            if sum_count != 0:
                transition_matrix[state1][state2] = transition_matrix[state1][state2] / sum_count

    # convert to mdp:
    # here we use the dot format to represent the mdp
    '''
    digraph learned_mdp {
    s0 [label="init"];
    s1 [label="pull-up-account"];
    s2 [label="end"];
    s3 [label="enter-details"];
    s0 -> s1  [label="pull-up-account:0.03125"];
    s0 -> s3  [label="enter-details:0.03125"];
    s0 -> s2  [label="end:0.03125"];
    s1 -> s3  [label="pull-up-account:0.2202970297029703"];
    s1 -> s1  [label="pull-up-account:0.7797029702970297"];
    s1 -> s3  [label="enter-details:0.03125"];
    s1 -> s2  [label="end:0.03125"];
    s2 -> s1  [label="pull-up-account:0.03125"];
    s2 -> s3  [label="enter-details:0.03125"];
    s2 -> s2  [label="end:0.03125"];
    s3 -> s1  [label="pull-up-account:0.03125"];
    s3 -> s3  [label="enter-details:0.03125"];
    s3 -> s2  [label="end:0.03125"];
    __start0 [label="", shape=none];
    __start0 -> s0  [label=""];
    }
    '''

    # ['find_hotel', 'book_hotel', 'find_train', 'book_train', 'find_attraction', 'find_restaurant', 'book_restaurant', 'find_hospital', 'book_taxi', 'find_taxi', 'find_bus', 'find_police']

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

    dot_str = convert_to_dot(transition_matrix)

    # write the dot string to a file
    with open(f"learned_mdp_abcd_flow_{data_part}_{flow_name}.dot", "w") as f:
        f.write(dot_str)

end = time.time()
print("time: ", end - start) 

# load an automaton
# automaton = load_automaton_from_file(f"learned_mdp_abcd_flow_{data_part}_{flow_name}.dot", automaton_type='mdp')

# visualize the automaton
# visualize_automaton(automaton, path=f"learned_mdp_abcd_flow_{data_part}_{flow_name}.png")
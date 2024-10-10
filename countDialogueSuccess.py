import os
import json
import jsonlines
# from bert_score import score
# from nltk.corpus import stopwords
import re
import numpy as np

abcd_woCP_path = "/research/d5/gds/xywen22/project/llm_framework/guidedTOD/abcd/dialogues/abcdASTWOActionFlowAll_wo_chainedPrior.json"
abcd_wCP_path = "/research/d5/gds/xywen22/project/llm_framework/guidedTOD/abcd/dialogues/abcdASTWOActionFlowAll_with_chainedPrior.json"

def compute_DSR(action_labels, action_preds, value_labels, value_preds, convo_ids, turn_ids):

    action_labels_arrary = np.array(action_labels, dtype=object)
    action_preds_arrary = np.array(action_preds, dtype=object)
    action_match = action_labels_arrary == action_preds_arrary

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

    print(f"action_acc: {action_acc}, value_acc: {value_acc}, joint_acc: {joint_acc}")

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
    dialogue_step_successes = []
    for convo_id, itm in conversations.items():
        # print(f"convo_id: {convo_id}")
        convo_correctness = itm[0]
        convo_correctness_action = itm[1]
        convo_correctness_value = itm[2]

        # print(f"convo_id: {convo_id}, convo_correctness: {convo_correctness}")
        def count_successive_ones(lst):
            count = 0
            for num in lst:
                if num:
                    count += 1
                else:
                    break
            return count
        successive_steps = count_successive_ones(convo_correctness)
        dialogue_step_successes.append(successive_steps/len(convo_correctness))

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

    em_action_score = sum(em_action) / float(len(em_action))
    em_value_score = sum(em_value) / float(len(em_value))
    em_joint_score = sum(em_joint) / float(len(em_joint))

    step_success_rate = sum(dialogue_step_successes) / len(dialogue_step_successes)

    return {
        "EM_action": round(em_action_score, 4),
        "EM_value": round(em_value_score, 4),
        "EM_joint": round(em_joint_score, 4),
        # "turn_acc_joint": round(turn_acc, 4),
        # "turn_acc_action": round(turn_acc_action, 4),
        # "turn_acc_value": round(turn_acc_value, 4),
        # "CE_joint": round(final_score, 4),
        # "CE_action": round(final_score_action, 4),
        # "CE_value": round(final_score_value, 4),
        "step_success_rate": round(step_success_rate, 4),
        "dialogue_success_rate": round(em_joint_score, 4)
    }

def parse_ast_prediction(prediction_str):
    # possibleActions = ['search-faq', 'search-timing', 'pull-up-account', 'verify-identity', 'membership', 'ask-the-oracle', 'search-shirt', 'search-policy', 'select-faq', 'send-link', 'enter-details', 'log-out-in', 'promo-code', 'notify-team', 'make-purchase', 'validate-purchase', 'update-order', 'subscription-status', 'make-password', 'try-again', 'shipping-status', 'record-reason', 'update-account', 'instructions', 'search-boots', 'search-jeans', 'search-membership', 'search-jacket', 'search-pricing', 'offer-refund', 'MISSING']
    match = re.match(r"(.*)\[(.*)]", prediction_str)
    # print(f"prediction_str: {prediction_str}, match: {match}")
    if match:
    # match = re.match(r"(.*)\[(.*)]", prediction_str)
    # if match:
        # action w/ value
        action_name = match.group(1).strip()
        slot_str = match.group(2)
        slot_str = slot_str.replace(";", ",")
        slots = [s.strip() for s in slot_str.split(",")]
        '''
        added by xiangyu
        '''
        # for i in range(len(slots)):
        #     if slots[i].endswith(">") and not slots[i].startswith("<"):
        #         # add "<" to the beginning of the slot
        #         slots[i] = "<" + slots[i]
        #     if slots[i].startswith("<") and not slots[i].endswith(">"):
        #         # add ">" to the end of the slot
        #         slots[i] = slots[i] + ">"
    else:
        action_name = "MISSING"
        slots = ["MISSING"]

    return action_name, slots

with jsonlines.open(abcd_woCP_path) as reader:
    action_preds = []
    action_labels = []

    value_preds = []
    value_labels = []
    convo_ids = []
    turn_ids = []
    for obj in reader:
        # print(obj)
        sample_id = obj.get('sample_id')
        convo_id = obj.get('convo_id')
        turn_id = obj.get('turn_id')
        target = obj.get('target')
        input = obj.get('input')
        pred_action = obj.get('pred_action')
        pred_utterance = obj.get('pred_utterance')
        label_utterance = obj.get('label_utterance')

        '''
        The target: search-faq [none]
        The pred_action: search-faq [none]
        '''

        tmp_action_label, tmp_value_label = parse_ast_prediction(target)
        tmp_action_pred, tmp_value_pred = parse_ast_prediction(pred_action)

        action_preds.append(tmp_action_pred)
        action_labels.append(tmp_action_label)
        value_preds.append(tmp_value_pred)
        value_labels.append(tmp_value_label)
        convo_ids.append(convo_id)
        turn_ids.append(turn_id)

    result = compute_DSR(action_labels, action_preds, value_labels, value_preds, convo_ids, turn_ids)

    # print(f"results: {result}")

    print(json.dumps(result, indent=4))


# with jsonlines.open(abcd_wCP_path) as reader:
#     for obj in reader:
#         print(obj)


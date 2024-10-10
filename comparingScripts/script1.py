def compute_ast_acc_metrics_beam_wMultiChain(predictions, labels, convo_ids, turn_ids, sequence_scores=None, num_beams=4):
    print("len(predictions): ", len(predictions))
    print("len(labels): ", len(labels))
    print("len(sequence_scores): ", len(sequence_scores))
    from aalpy.utils import load_automaton_from_file

    action_mapping = {
        "init": "init",
        "find_hotel": "search for hotel",
        "book_hotel": "book hotel",
        "find_train": "search for trains",
        "book_train": "book train ticket",
        "find_attraction": "search for attractions",
        "find_restaurant": "search for restaurants",
        "book_restaurant": "book table at restaurant",
        "find_hospital": "search for hospital",
        "book_taxi": "book taxi",
        "find_taxi": "search for taxi",
        "find_bus": "search for bus",
        "find_police": "search for police station"
    }

    action_mapping_reverse = {v: k for k, v in action_mapping.items()}
    # print(action_mapping_reverse)

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
    # print(f"state_mapping: {state_mapping}")

    transition_mapping_global = {}
    for transition in automaton_transitions:
        transition_split = transition.split('->')
        source_state = transition_split[0].strip()
        target_state = transition_split[1].strip().split(' ')[0]
        transition_label = transition_split[1].split('[label="')[1].split('"];')[0]
        transition_action = transition_label.split(':')[0]
        transition_freq = np.log(float(transition_label.split(':')[1])) if float(transition_label.split(':')[1]) > 0 else -10000
        transition_mapping_global[(state_mapping[source_state], state_mapping[target_state])] = (transition_action, transition_freq)

    # from s0 to s31, if some pair of states are not in the transition_mapping, then the frequency is 0
    for i in range(14):
        for j in range(14):
            if (state_mapping[f's{i}'], state_mapping[f's{j}']) not in transition_mapping_global:
                transition_mapping_global[(state_mapping[f's{i}'], state_mapping[f's{j}'])] = ('unknown', -10000)

    # group predictions every 4
    new_predictions = []
    for i in range(0, len(predictions), num_beams):
        new_predictions.append(predictions[i:i+num_beams])
        # new_predictions.append(predictions[i])
    
    new_sequence_scores = []
    for i in range(0, len(sequence_scores), num_beams):
        # print(f"sequence_scores[i:i+4]: {sequence_scores[i:i+4]}")
        # do no use normalization
        new_sequence_scores.append(sequence_scores[i:i+num_beams])

    
    previous_actions = ['init']
    current_convo_id = 999999
    new_new_predictions = []
    for new_pred, label1, new_sequence_score, convo_id1, turn_id1 in zip(new_predictions, labels, new_sequence_scores, convo_ids, turn_ids):
        if convo_id1 != current_convo_id:
            previous_actions = ['init']
            current_convo_id = convo_id1
        
        actions = []
        for pred in new_pred:
            action_value = pred.split(';')[0].strip()
            action = action_value.split(' [')[0].strip()
            actions.append(action)
        
        # action1 = new_pred[0].split(' ')[0].strip()
        # action2 = new_pred[1].split(' ')[0].strip()
        # action3 = new_pred[2].split(' ')[0].strip()
        # action4 = new_pred[3].split(' ')[0].strip()

        rates = []
        for i in range(len(actions)):
            try:
                # rate = transition_mapping_global[(previous_actions[-1], actions[i])][1]
                rate = transition_mapping_global[(action_mapping_reverse[previous_actions[-1]], action_mapping_reverse[actions[i]])][1]
            except:
                print("An error occurred while accessing the transition mapping.")
                rate = -10000
            rates.append(rate)

        '''
        the way to merge the two modules for post processng, v1
        '''
        exp_new_sequence_score = [np.exp(score) for score in new_sequence_score]
        exp_rates = [np.exp(rate) for rate in rates]
        norm_exp_new_sequence_score = exp_new_sequence_score / np.sum(exp_new_sequence_score)
        norm_exp_rates = exp_rates / np.sum(exp_rates)
        log_norm_exp_new_sequence_score = [np.log(score) for score in norm_exp_new_sequence_score]
        log_norm_exp_rates = [np.log(rate) for rate in norm_exp_rates]

        merge_scores = 0.9*np.array(log_norm_exp_new_sequence_score) + 0.1*np.array(log_norm_exp_rates)
        # merge_scores = log_norm_exp_new_sequence_score
        # merge_scores = rates

        max_index = np.argmax(merge_scores)
        new_new_predictions.append(new_pred[max_index])

        action_value_label = label1.split(';')[0].strip()
        action_label = action_value_label.split(' [')[0].strip()
        # action_label = action_mapping_reverse[action_label]
        previous_actions.append(action_label)

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
    # for pred, label in zip(predictions, labels):
        tmp_pred = pred.split(";")[0]
        tmp_label = label.split(";")[0]
        # print(f"pred: {pred}, label: {label}")
        action_label, values_label = parse_ast_prediction(tmp_label)
        values_label.sort()
        # for value in values_label:
        #     action_labels.append(action_label)
        #     value_labels.append(value)
        action_labels.append(action_label)
        value_labels.append(values_label)

        action_pred, values_pred = parse_ast_prediction(tmp_pred)
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
        
        # print(f"convo_id: {convo_id}, convo_correctness: {convo_correctness}")
        current_score = 0
        convo_length = len(convo_correctness)
        # we use turn_id rather than the true turn_count since turn counts will skip numbers
        # when looping through the conversation due to skipping over customer utterances

        snipet_lens = [1,2,3]

        # for joint correctness
        snipet_lens_joint  = snipet_lens
        snipet_numbers_joint = [0] * len(snipet_lens_joint)
        snipet_correct_joint = [0] * len(snipet_lens_joint)
        # for each dialogue, compute the rate of each length of snipet that is correct, using the sliding window of the length
        for snipet_i in range(len(snipet_lens_joint)):
            # print("convo_length: ", convo_length)
            if snipet_lens_joint[snipet_i] > convo_length:
                continue
            # print(f"snipet_i: {snipet_i}")
            snipet_len = snipet_lens_joint[snipet_i]
            for turn_id in range(convo_length - snipet_len + 1):
                snipet_numbers_joint[snipet_i] += 1
                if sum(convo_correctness[turn_id:turn_id+snipet_len]) == snipet_len:
                    snipet_correct_joint[snipet_i] += 1

        average_counter = 0
        for snipet_i in range(len(snipet_lens_joint)):
            # print(f"snipet_correct_joint: {snipet_correct_joint[snipet_i]}, snipet_numbers_joint: {snipet_numbers_joint[snipet_i]}")
            if snipet_numbers_joint[snipet_i] == 0:
                continue
            snipet_correct_joint[snipet_i] = snipet_correct_joint[snipet_i] / snipet_numbers_joint[snipet_i]
            average_counter += 1
        
        # print(f"snipet_correct: {snipet_correct_joint}")
        # print("average_counter: ", average_counter)
        average_for_dialogue = 0
        for snipet_i in range(len(snipet_lens_joint)):
            average_for_dialogue += snipet_correct_joint[snipet_i]
        average_for_dialogue = average_for_dialogue / min(len(snipet_lens), len(convo_correctness_action))
        # average_for_dialogue = average_for_dialogue / average_counter
        # print(f"average_for_dialogue: {average_for_dialogue}")

        turn_score += average_for_dialogue

        # for action correctness
        snipet_lens_action  = snipet_lens
        snipet_numbers_action = [0] * len(snipet_lens_action)
        snipet_correct_action = [0] * len(snipet_lens_action)
        # for each dialogue, compute the rate of each length of snipet that is correct, using the sliding window of the length
        for snipet_i in range(len(snipet_lens_action)):
            # print("convo_length: ", convo_length)
            if snipet_lens_action[snipet_i] > convo_length:
                continue
            # print(f"snipet_i: {snipet_i}")
            snipet_len = snipet_lens_action[snipet_i]
            for turn_id in range(convo_length - snipet_len + 1):
                snipet_numbers_action[snipet_i] += 1
                if sum(convo_correctness_action[turn_id:turn_id+snipet_len]) == snipet_len:
                    snipet_correct_action[snipet_i] += 1

        average_counter = 0
        for snipet_i in range(len(snipet_lens_action)):
            if snipet_numbers_action[snipet_i] == 0:
                continue
            snipet_correct_action[snipet_i] = snipet_correct_action[snipet_i] / snipet_numbers_action[snipet_i]
            average_counter += 1

        # print(f"snipet_correct: {snipet_correct_action}")
        average_for_dialogue = 0
        for snipet_i in range(len(snipet_lens_action)):
            average_for_dialogue += snipet_correct_action[snipet_i]
        average_for_dialogue = average_for_dialogue / min(len(snipet_lens), len(convo_correctness_action))
        # average_for_dialogue = average_for_dialogue / average_counter
        # print(f"average_for_dialogue: {average_for_dialogue}")

        turn_score_action += average_for_dialogue

        # for value correctness
        snipet_lens_value  = snipet_lens
        snipet_numbers_value = [0] * len(snipet_lens_value)
        snipet_correct_value = [0] * len(snipet_lens_value)
        # for each dialogue, compute the rate of each length of snipet that is correct, using the sliding window of the length
        for snipet_i in range(len(snipet_lens_value)):
            # print("convo_length: ", convo_length)
            if snipet_lens_value[snipet_i] > convo_length:
                continue
            # print(f"snipet_i: {snipet_i}")
            snipet_len = snipet_lens_value[snipet_i]
            for turn_id in range(convo_length - snipet_len + 1):
                snipet_numbers_value[snipet_i] += 1
                if sum(convo_correctness_value[turn_id:turn_id+snipet_len]) == snipet_len:
                    snipet_correct_value[snipet_i] += 1

        average_counter = 0
        for snipet_i in range(len(snipet_lens_value)):
            if snipet_numbers_value[snipet_i] == 0:
                continue
            snipet_correct_value[snipet_i] = snipet_correct_value[snipet_i] / snipet_numbers_value[snipet_i]
            average_counter += 1

        # print(f"snipet_correct: {snipet_correct_value}")
        average_for_dialogue = 0
        for snipet_i in range(len(snipet_lens_value)):
            average_for_dialogue += snipet_correct_value[snipet_i]
        average_for_dialogue = average_for_dialogue / min(len(snipet_lens), len(convo_correctness_action))
        # average_for_dialogue = average_for_dialogue / average_counter
        # print(f"average_for_dialogue: {average_for_dialogue}")

        turn_score_value += average_for_dialogue

    # normalize by total number of turns possible
    '''
    len(convo_ids): 200, len(turn_ids): 200
    '''
    # print(f"len(convo_ids): {len(convo_ids)}, len(turn_ids): {len(turn_ids)}")
    turn_acc = turn_correct / float(len(conversations))
    turn_acc_action = turn_correct_action / float(len(conversations))
    turn_acc_value = turn_correct_value / float(len(conversations))
    final_score = turn_score / float(len(conversations))
    final_score_action = turn_score_action / float(len(conversations))
    final_score_value = turn_score_value / float(len(conversations))
    
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
        "CE_joint": round(final_score, 4),
        "CE_action": round(final_score_action, 4),
        "CE_value": round(final_score_value, 4),
        "step_success_rate": round(step_success_rate, 4),
        "dialogue_success_rate": round(em_joint_score, 4)
    }

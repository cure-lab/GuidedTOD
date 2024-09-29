import json
import os

dialogues = {}
with open("data/processed/incremental_data.json", "r") as f:
    lines = f.readlines()
    tmp_dialogue = []
    for line in lines:
        data = json.loads(line)
        if data["convo_id"] not in dialogues:
            dialogues[data["convo_id"]] = []
        dialogues[data["convo_id"]].append(data)
    

print(dialogues.keys())
print(len(dialogues.keys()))

# load original dialogues
original_dialogues = []
with open("data/processed/train_AST_abcd_10p.json", "r") as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        original_dialogues.append(data)

# get max sample id from original dialogues
max_sample_id = 0
for d in original_dialogues:
    if d["sample_id"] > max_sample_id:
        max_sample_id = d["sample_id"]
print(max_sample_id)


# select first 100 dialogues from dialogues and append to original dialogues
counter = 0
for i, (k, v) in enumerate(dialogues.items()):
    if i >= 100:
        break
    # original_dialogues.append(v)
    for d in v:
        d["sample_id"] = max_sample_id + counter + 1
        d["target"] = d["predicted_action"]
        # remove "predicted_action" key
        tmp = d.copy()
        tmp.pop("predicted_action")
        original_dialogues.append(tmp)
        counter += 1

# check if data/processed/train_AST_abcd_10p_updating100.json exists
if os.path.exists("data/processed/train_AST_abcd_10p_updating100.json"):
    os.remove("data/processed/train_AST_abcd_10p_updating100.json")
# save to file data/processed/train_AST_abcd_10p_updating100.json
with open("data/processed/train_AST_abcd_10p_updating100.json", "w") as f:
    for d in original_dialogues:
        f.write(json.dumps(d))
        f.write("\n")

# select first 200 dialogues from dialogues and append to original dialogues
counter = 0
for i, (k, v) in enumerate(dialogues.items()):
    if i >= 200:
        break
    # original_dialogues.append(v)
    for d in v:
        d["sample_id"] = max_sample_id + counter + 1
        d["target"] = d["predicted_action"]
        # remove "predicted_action" key
        tmp = d.copy()
        tmp.pop("predicted_action")
        original_dialogues.append(tmp)
        counter += 1

# check if data/processed/train_AST_abcd_10p_updating200.json exists
if os.path.exists("data/processed/train_AST_abcd_10p_updating200.json"):
    os.remove("data/processed/train_AST_abcd_10p_updating200.json")
# save to file data/processed/train_AST_abcd_10p_updating200.json
with open("data/processed/train_AST_abcd_10p_updating200.json", "w") as f:
    for d in original_dialogues:
        f.write(json.dumps(d))
        f.write("\n")

# select first 300 dialogues from dialogues and append to original dialogues
counter = 0
for i, (k, v) in enumerate(dialogues.items()):
    if i >= 300:
        break
    # original_dialogues.append(v)
    for d in v:
        d["sample_id"] = max_sample_id + counter + 1
        d["target"] = d["predicted_action"]
        # remove "predicted_action" key
        tmp = d.copy()
        tmp.pop("predicted_action")
        original_dialogues.append(tmp)
        counter += 1

# check if data/processed/train_AST_abcd_10p_updating300.json exists
if os.path.exists("data/processed/train_AST_abcd_10p_updating300.json"):
    os.remove("data/processed/train_AST_abcd_10p_updating300.json")
# save to file data/processed/train_AST_abcd_10p_updating300.json
with open("data/processed/train_AST_abcd_10p_updating300.json", "w") as f:
    for d in original_dialogues:
        f.write(json.dumps(d))
        f.write("\n")

# select first 400 dialogues from dialogues and append to original dialogues
counter = 0
for i, (k, v) in enumerate(dialogues.items()):
    if i >= 400:
        break
    # original_dialogues.append(v)
    for d in v:
        d["sample_id"] = max_sample_id + counter + 1
        d["target"] = d["predicted_action"]
        # remove "predicted_action" key
        tmp = d.copy()
        tmp.pop("predicted_action")
        original_dialogues.append(tmp)
        counter += 1

# check if data/processed/train_AST_abcd_10p_updating400.json exists
if os.path.exists("data/processed/train_AST_abcd_10p_updating400.json"):
    os.remove("data/processed/train_AST_abcd_10p_updating400.json")
# save to file data/processed/train_AST_abcd_10p_updating400.json
with open("data/processed/train_AST_abcd_10p_updating400.json", "w") as f:
    for d in original_dialogues:
        f.write(json.dumps(d))
        f.write("\n")

# select first 500 dialogues from dialogues and append to original dialogues
counter = 0
for i, (k, v) in enumerate(dialogues.items()):
    if i >= 500:
        break
    # original_dialogues.append(v)
    for d in v:
        d["sample_id"] = max_sample_id + counter + 1
        d["target"] = d["predicted_action"]
        # remove "predicted_action" key
        tmp = d.copy()
        tmp.pop("predicted_action")
        original_dialogues.append(tmp)
        counter += 1

# check if data/processed/train_AST_abcd_10p_updating500.json exists
if os.path.exists("data/processed/train_AST_abcd_10p_updating500.json"):
    os.remove("data/processed/train_AST_abcd_10p_updating500.json")
# save to file data/processed/train_AST_abcd_10p_updating500.json
with open("data/processed/train_AST_abcd_10p_updating500.json", "w") as f:
    for d in original_dialogues:
        f.write(json.dumps(d))
        f.write("\n")
import jsonlines

path = 'data/raw/test_multiwoz_22.json'

services = []

with jsonlines.open(path) as reader:
    for obj in reader:
        services.append(obj['services'])    

# select the top 5 dialogues that have the most services
top_5_index = sorted(range(len(services)), key=lambda i: len(services[i]), reverse=True)[:5]

top_5_dialogues = []
with jsonlines.open(path) as reader:
    for i, obj in enumerate(reader):
        if i in top_5_index:
            top_5_dialogues.append(obj['turns']['utterance'])

for dialogue in top_5_dialogues:
    print(dialogue)
    print('\n')
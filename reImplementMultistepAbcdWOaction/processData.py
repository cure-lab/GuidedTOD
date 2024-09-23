import json
import os

# read the json file line by line
def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':
    # read the json file
    dataSource = read_json_file('/research/d5/gds/xywen22/project/llm_framework/workflow-discovery-abcd-w-action/data/processed/test_AST_abcd.json')
    dataTargetFormat = read_json_file('/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd-full.json')
    '''
    dataSource: 
    {"sample_id": 0, "target": "search-faq [none]", "input": "Context: hello. how can i help you today? hi.  my name is chloe zhang.  i am curious as to when my promo code expires. would you be able to tell me? yes let me look into this im sure we can find a solution", "target_data": "[\"search-faq\", [\"none\"]]"}
    '''
    '''
    dataTargetFormat:
    {"sample_id": 0, "convo_id": 4989, "turn_id": 5, "target": "search-faq [none]", "input": "Context: hello. how can i help you today? hi.  my name is chloe zhang.  i am curious as to when my promo code expires. would you be able to tell me? yes let me look into this im sure we can find a solution", "target_data": "[\"search-faq\", [\"none\"]]"}
    '''
    print(len(dataSource))
    print(len(dataTargetFormat))
    for i in range(len(dataSource)):
        if dataSource[i]['sample_id'] != dataTargetFormat[i]['sample_id']:
            print('sample_id not equal')
        if dataSource[i]['target'] != dataTargetFormat[i]['target']:
            print('target not equal')
        dataSource[i]['convo_id'] = dataTargetFormat[i]['convo_id']
        dataSource[i]['turn_id'] = dataTargetFormat[i]['turn_id']
    
    # write the json file
    with open('/research/d5/gds/xywen22/project/llm_framework/AST_abcd_part/data/processed/test_AST_abcd_w_action_full.json', 'w') as f:
        for line in dataSource:
            f.write(json.dumps(line) + '\n')

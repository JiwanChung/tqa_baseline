import pickle
import json
from collections import Counter
import argparse
import re
from collections import OrderedDict

parser = argparse.ArgumentParser(description='TQA basline Text model')
parser.add_argument('--ckpt-name', '-c', default='', help='type of file to process')
args = parser.parse_args()

source_dir = 'prepro/data/'

test_dict = pickle.load(open(source_dir + 'correct_dict_val{}.pickle'.format(args.ckpt_name), 'r'))

data_path = '/home/jiwan/tqa/data/val/tqa_v1_val.json'

data = json.load(open(data_path, 'r'))

right_counter = Counter()
wrong_counter = Counter()

output_right = []
output_wrong = []
for lesson in data:
    for q in lesson['questions']['nonDiagramQuestions'].values():
        id = q['globalID']
        if id in test_dict:
            item = test_dict[id]
            if item[0] != item[1]:
                if 'questionSubType' in q:
                    wrong_counter[q['questionSubType']] += 1
                output_wrong.append([id, item, q['answerChoices'], q['beingAsked']])
            else:
                output_right.append([id, item, q['answerChoices'], q['beingAsked']])
                if 'questionSubType' in q:
                    right_counter[q['questionSubType']] += 1

def keys(x):
    xid = x[0]
    xid = re.sub(r'[A-Za-z]*_', r'', xid)
    return int(xid)

output_right.sort(key=keys)
output_wrong.sort(key=keys)
print('sorted')
dicts = [OrderedDict((a[0], a) for a in output_right), OrderedDict((a[0], a) for a in output_wrong)]

with open('prepro/data/wrong_answer_pair{}.json'.format(args.ckpt_name), 'w') as outfile:
    json.dump(dicts[1], outfile)

with open('prepro/data/right_answer_pair{}.json'.format(args.ckpt_name), 'w') as outfile:
    json.dump(dicts[0], outfile)

with open('prepro/data/stats{}.json'.format(args.ckpt_name), 'w') as outfile:
    json.dump("Right:{}, Wrong:{}".format(right_counter, wrong_counter), outfile)

# -*- coding: utf-8 -*-

import argparse
import json
import os
import pandas as pd
import csv
import re
from collections import OrderedDict
import sys

sys.path.append(os.path.join(os.path.expanduser('~'), 'mc', 'utils'))
from tokenizer import Tokenizer

from tqdm import tqdm

def main():
    args = get_args()
    prepro(args)

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    base_dir = os.path.join(home, 'tqa')
    source_dir = os.path.join(base_dir, "data")
    target_dir = os.path.join(base_dir, 'prepro','data')

    parser.add_argument('-b', "--base_dir", default=base_dir)
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument('-top', "--single-topic", action='store_true') # store_true -> False!!
    parser.add_argument('-p', "--if-pair", action='store_true')
    parser.add_argument('-dq', "--diagram_questions", action='store_true')
    parser.add_argument('-to', "--tokenizer", default='spacy')
    
    # TODO : put more args here
    return parser.parse_args()

def remove_delim(word):
    word = re.sub(r"\.", "", word)
    return word

def prepro(args):
    prepro_each(args, 'train')
    prepro_each(args, 'test')
    prepro_each(args, 'val')

def prepro_each(args, data_type):
    # open
    source_path = os.path.join(args.source_dir, "{}/tqa_v1_{}.json".format(data_type, data_type))
    source_data = json.load(open(source_path, 'r'))

    # tabularize
    tabular = []
    pair = []

    stat = {}
    stat['max_topic_words'] = 0
    stat['max_question_words'] = 0
    stat['max_answer_words'] = 0
    stat['max_answers'] = 0
    stat['max_topics'] = 0

    tokenizer = Tokenizer(args.tokenizer)
    tokenize = tokenizer.tokenize

    for index_l, lesson in enumerate(tqdm(source_data)):
        topics = []
        stat['max_topics'] = max(stat['max_topics'], len(lesson['topics']))
        for key_t, topic in lesson['topics'].items():
            stat['max_topic_words']  = max(stat['max_topic_words'], len(tokenize(topic)))
            topics.append(topic['content']['text'])

        def getqs(q_raw_data, with_diagram=False, if_pair=False, stat=None):
            q_tabular = []

            for key_q, q in q_raw_data.items():
                question = q['beingAsked']['processedText']

                stat['max_question_words'] = max(stat['max_question_words'], len(tokenize(question)))

                # handling weird exception of questions without any correct answer
                # WARNING: obviously this rules out the entire test set effectively
                if 'correctAnswer' not in q:
                    #print(q['globalID'])
                    continue

                correct_answer = q['correctAnswer']['processedText']

                ans_sorted = OrderedDict(sorted(q['answerChoices'].items()))

                answers = []
                correct_index = -1

                for index_a, (key_a, answer) in enumerate(ans_sorted.iteritems()):
                    answer_str = answer['processedText']
                    idstruct = answer['idStructural']
                    answers.append(answer_str)

                    stat['max_answer_words'] = max(stat['max_answer_words'], len(tokenize(answer_str)))

                    if answer_str == correct_answer or tokenize(correct_answer) == tokenize(remove_delim(idstruct)):
                        correct_index = index_a

                stat['max_answers'] = max(stat['max_answers'], len(answers))

                ltos = listToString()

                relev_topic = ltos.run(topics)

                # TODO: process diagrams to CNN features

                # ignore data if the question does not have a valid answer
                if correct_index >= 0:
                    correct_answer = answers[correct_index]
                    answer_string = ltos.run(answers)
                    if if_pair:
                        for ans in answers:
                            q_tabular.append({'question':question, 'correct_answer':correct_answer, 'wrong_answer':ans, 'topic':relev_topic, 'id': q['globalID']})
                    else:
                        q_tabular.append({'question':question, 'correct_answer':correct_index, 'answers':answer_string, 'topic':relev_topic, 'id': q['globalID']})

            return q_tabular

        nondq_raw = lesson['questions']['nonDiagramQuestions']

        nondq_tab = getqs(nondq_raw, False, False, stat)
        if args.if_pair:
            nondq_pair = getqs(nondq_raw, False, True, stat)
            pair += nondq_pair

        tabular += nondq_tab
        '''if args.diagram_questions:
            dq_tab = getqs(nondq_raw, False, stat)
            dq_raw = lesson['questions']['diagramQuestions']
            tabular += dq_tab'''

    # save
    stats = {u'topic_size': stat['max_topic_words'], u'topic_num': stat['max_topics'], u'question_size': stat['max_question_words'], u'answer_size': stat['max_answers'], u'answer_num': stat['max_answer_words']}
    save(args, tabular, data_type, stats)
    if args.if_pair:
        save(args, pair, '{}_{}'.format(data_type, 'pair'), stats)

class listToString():
    def __init__(self):
        self.regex_quote = re.compile(r'\'*')
        self.regex_doublequote = re.compile(r'\"*')
    def run(self, list_data):
        return '[' + ''.join('"{}", '.format(self.regex_quote.sub(r'', self.regex_doublequote.sub(r'', datum))) for datum in list_data)[:-2] + ']'

def save(args, data, data_type, stats):
    add_opt = ''
    if not args.single_topic:
        add_opt = '_full'
    data_path = os.path.join(args.target_dir, "data_{}{}.tsv".format(data_type, add_opt))
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)

    with open(os.path.join(args.target_dir, "stat_{}{}.json".format(data_type, add_opt)), 'w') as file:
        json.dump(stats, file)

if __name__ == "__main__":
    main()

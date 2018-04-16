import argparse
import json
import os
import pandas as pd
from math import log as ln
import spacy
import re
from collections import OrderedDict

from tqdm import tqdm

nlp = spacy.load('en')

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

    # TODO : put more args here
    return parser.parse_args()

def remove_delim(word):
    word = re.sub(r"\.", "", word)
    return word

def token_sent(sent):

    sent = re.sub(
        r"[\*\"\n\\\+\-\/\=\(\):\[\]\|\!;]", " ",
        str(sent))
    sent = re.sub(r"[ ]+", " ", sent)
    sent = re.sub(r"\!+", "!", sent)
    sent = re.sub(r"\,+", ",", sent)
    sent = re.sub(r"\?+", "?", sent)

    sent = unicode(sent)
    return [x.text for x in nlp.tokenizer(sent) if x.text != " "]

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

    for index_l, lesson in enumerate(tqdm(source_data)):
        topics = []
        for key_t, topic in lesson['topics'].items():
            topics.append(topic['content']['text'])

        def getqs(q_raw_data, with_diagram=False, if_pair=False):
            q_tabular = []

            for key_q, q in q_raw_data.items():
                question = q['beingAsked']['processedText']

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
                    if answer_str == correct_answer or token_sent(correct_answer) == token_sent(remove_delim(idstruct)):
                        correct_index = index_a

                if args.single_topic:
                    def get_idf(word, docs):
                        N = len(docs)
                        count = 0
                        for doc in docs:
                            if word in doc:
                                count += 1

                        return ln(float(N)/(1+count))

                    def get_tf(word, doc):
                        N = len(doc)
                        count = doc.count(word)

                        return float(count)/N

                    # pick relev topic based on tf-idf
                    idf = {}

                    for word in question:
                        idf[word] = get_idf(word, topics)

                    max_score = 0
                    max_arg = 0
                    for index_topic, topic in enumerate(topics):
                        if len(topic) > 0:
                            score_topic = 0
                            for word in question:
                                score_topic += get_tf(word, topic)*idf[word]
                            if score_topic > max_score:
                                max_score = score_topic
                                max_arg = index_topic

                    relev_topic = topics[max_arg]
                else:
                    relev_topic = topics

                # TODO: process diagrams to CNN features

                # ignore data if the question does not have a valid answer
                if correct_index >= 0:

                    if data_type == 'train':
                        correct_answer = answers[correct_index]
                        if if_pair:
                            for ans in answers:
                                q_tabular.append({'question':question, 'correct_answer':correct_answer, 'wrong_answer':ans, 'topic':relev_topic, 'id': q['globalID']})
                        else:
                            q_tabular.append({'question':question, 'correct_answer':correct_index, 'answers':answers, 'topic':relev_topic, 'id': q['globalID']})
                    else:
                        correct_answer = answers[correct_index]
                        if if_pair:
                            for ans in answers:
                                q_tabular.append({'question':question, 'correct_answer':correct_answer, 'wrong_answer':ans, 'topic':relev_topic, 'id': q['globalID']})
                        else:
                            q_tabular.append({'question':question, 'correct_answer':correct_index, 'answers':answers, 'topic':relev_topic, 'id': q['globalID']})

            return q_tabular

        nondq_raw = lesson['questions']['nonDiagramQuestions']

        nondq_tab = getqs(nondq_raw, False, False)
        if args.if_pair:
            nondq_pair = getqs(nondq_raw, False, True)
            pair += nondq_pair

        tabular += nondq_tab
        '''if args.diagram_questions:
            dq_tab = getqs(nondq_raw, False)
            dq_raw = lesson['questions']['diagramQuestions']
            tabular += dq_tab'''

    # save
    save(args, tabular, data_type)
    if args.if_pair:
        save(args, pair, '{}_{}'.format(data_type, 'pair'))

def save(args, data, data_type):
    add_opt = ''
    if not args.single_topic:
        add_opt = '_full'
    data_path = os.path.join(args.target_dir, "data_{}{}.tsv".format(data_type, add_opt))
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False, sep='\t')

if __name__ == "__main__":
    main()

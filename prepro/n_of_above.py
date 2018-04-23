import json
import re


def check_above(text):
    x = re.search(r'.*?of.*?above.*?', text)
    return (False if x is None else True)


def main(dtype):
    data_path = 'prepro/data/{}_answer_pair.json'.format(dtype)
    data = json.load(open(data_path, 'r'))

    output = []
    count = 0
    for q in data:
        answers = q[2]
        for ans in answers.values():
            if check_above(ans['processedText']):
                output.append(q)
                count += 1

    output = ['above:{}/ {}'.format(count, len(data))] + output
    with open('prepro/data/{}_n_of_above.json'.format(dtype), 'w') as outfile:
        json.dump(output, outfile)


main('right')
main('wrong')

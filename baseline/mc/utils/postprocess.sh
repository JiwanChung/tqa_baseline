cd ~/tqa
python /home/jiwan/tqa/prepro/postpro_test.py -c $1
python -m json.tool "/home/jiwan/tqa/prepro/data/right_answer_pair$1.json" > "pretty_right_answer_pair$1.json"
python -m json.tool "/home/jiwan/tqa/prepro/data/wrong_answer_pair$1.json" > "pretty_wrong_answer_pair$1.json"
cp "/home/jiwan/tqa/prepro/data/pretty_wrong_answer_pair$1.json" /home/jiwan/mc
cp "/home/jiwan/tqa/prepro/data/pretty_right_answer_pair$1.json" /home/jiwan/mc

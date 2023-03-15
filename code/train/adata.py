#
import json
import pandas as pd
from tqdm import tqdm
from paddlenlp.datasets import load_dataset

datap = "../xfdata/address/"
datan = "Xeon3NLP_round1"
# "Xeon3NLP_round1_train_20210524.txt"
# "Xeon3NLP_round1_test_20210524.txt"

r_result = []
with open(f"{datap}/{datan}_train_20210524.txt") as f:
    for i in tqdm(f):
        j = json.loads(i)
        
        for _j in j["candidate"]:
            r_result.append([j["query"], _j["text"], _j["label"]])
        
r_result = pd.DataFrame(r_result)
r_result = r_result[
    (~r_result[0].isna()) &
    (~r_result[1].isna()) &
    (r_result[2] != "部分匹配")
]
r_result[2] = r_result[2].apply(lambda x: 1 if x == "完全匹配" else 0)

print(r_result)
print(pd.value_counts(r_result[2]))

r_result.head(40000).to_csv("../user_data/cut_data/trainE3.csv", index=False, header=False, sep="\t")


def read(data_path):
    with open(data_path, 'r') as f:
        for line in f:
            query, title, label = line.strip('\n').split('\t')
            yield {'query': query, 'title': title, 'label': int(label)}

for i in load_dataset(read, data_path='../user_data/cut_data/trainE3.csv', lazy=False):
    print(i)
    break

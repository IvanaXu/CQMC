import json
import pandas as pd
from sklearn.model_selection import train_test_split
from paddlenlp.datasets import load_dataset
from sklearn.utils import shuffle


# 1/LCQMC
_ds1, _ds2, _ds3 = load_dataset("lcqmc")
print(len(_ds1), len(_ds2), len(_ds3))
d1 = [i for _ds in [_ds1, _ds2] for i in _ds]

# 2/AFQMC
d2 = []
with open("../xfdata/2/train.json", "r") as f:
    for i in f:
        d2.append(json.loads(i))
with open("../xfdata/2/dev.json", "r") as f:
    for i in f:
        d2.append(json.loads(i))

# 3/MOQMC 跨领域迁移的文本语义匹配
d3 = pd.read_csv("../xfdata/3/train.tsv", sep="\t", header=None)

# 4/C19QMC 新冠疫情相似问句判定数据集
d4 = pd.read_csv("../xfdata/4/test.csv", usecols=["id", "query1", "query2"])
d4l = pd.read_csv("../xfdata/4/test.label.csv", usecols=["id", "label"])
d4 = pd.merge(d4, d4l, on="id", how="inner")
del d4l

# 5/iflytek 中文对话文本匹配挑战赛
d5 = pd.read_csv("../xfdata/5/train.csv", sep="\t", header=None)

#
trainE1 = pd.concat([
    pd.DataFrame(d1).rename({"query": 0, "title": 1, "label": 2}, axis=1),
    pd.DataFrame(d2).rename({"sentence1": 0, "sentence2": 1, "label": 2}, axis=1),
    d3.rename({1: 0, 2: 1, 0: 2}, axis=1),
    d4[["query1", "query2", "label"]].rename({"query1": 0, "query2": 1, "label": 2}, axis=1),
    d5,
])
trainE1 = shuffle(trainE1, random_state=10086)

trainE1, trainE2 = train_test_split(
    trainE1,
    test_size=5000,
    random_state=930721,
)

print("E1 E2", trainE1.shape, trainE2.shape)
trainE1.to_csv("../user_data/cut_data/trainE1.csv", index=False, header=False, sep="\t")
trainE2.to_csv("../user_data/cut_data/trainE2.csv", index=False, header=False, sep="\t")


def read(data_path):
    with open(data_path, 'r') as f:
        for line in f:
            query, title, label = line.strip('\n').split('\t')
            yield {'query': query, 'title': title, 'label': int(label)}


#
for i in load_dataset(read, data_path='../user_data/cut_data/trainE1.csv', lazy=False):
    print(i)
    break

for i in load_dataset(read, data_path='../user_data/cut_data/trainE2.csv', lazy=False):
    print(i)
    break

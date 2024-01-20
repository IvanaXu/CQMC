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

# 6/千言数据集：文本相似度
d6_1 = pd.read_csv(
    "../xfdata/6/bq_corpus/train.tsv", sep="\t", header=None, 
    # on_bad_lines='skip',
    error_bad_lines=False,
)
d6_2 = pd.read_csv("../xfdata/6/bq_corpus/dev.tsv", sep="\t", header=None)
d6_3 = pd.read_csv("../xfdata/6/lcqmc/train.tsv", sep="\t", header=None)
d6_4 = pd.read_csv("../xfdata/6/lcqmc/dev.tsv", sep="\t", header=None)
d6_5 = pd.read_csv("../xfdata/6/paws-x-zh/train.tsv", sep="\t", header=None)
d6_6 = pd.read_csv("../xfdata/6/paws-x-zh/dev.tsv", sep="\t", header=None)
d6 = pd.concat([
    d6_1, d6_2, 
    d6_3, d6_4, 
    d6_5, d6_6
])
del d6_1, d6_2, d6_3, d6_4, d6_5, d6_6

# 7/BUSTM FewCLUE 评测中对话短文本语义匹配数据集 2分类任务
d7 = load_dataset("fewclue", "bustm")
d7 = pd.concat([pd.DataFrame(list(i)) for i in d7])
d7 = d7.rename({"sentence1": 0, "sentence2": 1, "label": 2}, axis=1)[[0, 1, 2]]

# 8/CMNLI 中文语言推理任务
# 中立 neutral(0), 蕴含 entailment(1), 矛盾 contradiction(2)
# 很多论文已经提出，使用矛盾文本对作为难负例训练模型，可以提高模型效果
_ds1, _ds2, _ds3 = load_dataset("clue", "cmnli")
print(len(_ds1), len(_ds2), len(_ds3))
d8 = [i for _ds in [_ds1, _ds2] for i in _ds]
d8 = pd.DataFrame(d8).rename({"sentence1": 0, "sentence2": 1, "label": 2}, axis=1)
d8[2] = d8[2].apply(lambda x: 1 if x == 1 else 0)




#
trainE0 = pd.concat([
    pd.DataFrame(d1).rename({"query": 0, "title": 1, "label": 2}, axis=1),
    pd.DataFrame(d2).rename({"sentence1": 0, "sentence2": 1, "label": 2}, axis=1),
    d3.rename({1: 0, 2: 1, 0: 2}, axis=1),
    d4[["query1", "query2", "label"]].rename({"query1": 0, "query2": 1, "label": 2}, axis=1),
    d5,
    d6,
    d7,
    d8,
])
print("E0", trainE0.shape)

print(pd.value_counts(trainE0[2], dropna=False))
trainE0 = trainE0[~trainE0[2].isna()]

trainE0[0] = ["/" if pd.isna(i) else str(i) for i in trainE0[0]]
trainE0[1] = ["/" if pd.isna(i) else str(i) for i in trainE0[1]]
trainE0[2] = trainE0[2].apply(int)
print(pd.value_counts(trainE0[2], dropna=False))
print("E0", trainE0.shape)

trainE0.reset_index(drop=True, inplace=True)
trainE0.drop_duplicates(
    inplace=False
)
print("E0 Drop_duplicates", trainE0.shape)
trainE0 = shuffle(trainE0, random_state=10086)


#
trainE1, trainE2 = train_test_split(
    trainE0,
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

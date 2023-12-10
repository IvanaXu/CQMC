#
import erniebot
import numpy as np
import pandas as pd
from config import KEY
from tqdm import tqdm
from sklearn.model_selection import train_test_split


#
erniebot.api_type = 'aistudio'
erniebot.access_token = KEY
print(f"KEY {KEY}")


#
L, N = [], 10

trainE1 = pd.read_csv("../user_data/cut_data/trainE1.csv", sep="\t", header=None).head(N)
trainE1["T"] = "trainE1"
L.append(trainE1)

trainE2 = pd.read_csv("../user_data/cut_data/trainE2.csv", sep="\t", header=None).head(N)
trainE2["T"] = "trainE2"
L.append(trainE2)

test3 = pd.read_csv("../xfdata/3/test.tsv", sep="\t", header=None).head(N)
test3[2] = -1
test3["T"] = "test3"
L.append(test3)

test5 = pd.read_csv("../xfdata/5/test.csv", sep="\t", header=None).head(N)
test5[2] = -1
test5["T"] = "test5"
L.append(test5)

for task in ["bq_corpus", "lcqmc", "paws-x-zh"]:
    test6 = pd.read_csv(f"../xfdata/6/{task}/test.tsv", sep="\t", header=None).head(N)
    test6[2] = -1
    test6["T"] = f"test6_{task}"
    L.append(test6)

data = pd.concat(L)
print(pd.value_counts(data["T"]))


def get_Embedding(df):
    _X, _Y, _T = [], [], []
    for x1, x2, y, t in tqdm(zip(df[0], df[1], df[2], df["T"]), total=len(df)):
        _X.append(f"{x1}\t{x2}")
        _Y.append(y)
        _T.append(t)

    response = erniebot.Embedding.create(
        model='ernie-text-embedding',
        input=_X,
    )

    data = pd.DataFrame(response.get_result())
    data["Y"] = _Y
    data["T"] = _T

    #
    for _type in [
        "trainE1",
        "trainE2",
        "test3",
        "test5",
        "test6_bq_corpus", "test6_lcqmc", "test6_paws-x-zh"
    ]:
        data[data["T"] == _type].to_csv(f"../user_data/cut_data/{_type}_EMB.csv", index=False)


get_Embedding(df=data)




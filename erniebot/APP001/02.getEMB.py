#
import os
import time
import hashlib
import pickle
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
L, N = [], None

trainE1 = pd.read_csv("../user_data/cut_data/trainE1.csv", sep="\t", header=None) # 1181605
print(f"trainE1 {trainE1.shape}")
trainE1["T"] = "trainE1"
_, trainE1 = train_test_split(trainE1, test_size=181605, random_state=10086)
L.append(trainE1)

trainE2 = pd.read_csv("../user_data/cut_data/trainE2.csv", sep="\t", header=None)
print(f"trainE2 {trainE2.shape}")
trainE2["T"] = "trainE2"
L.append(trainE2)

test3 = pd.read_csv("../xfdata/3/test.tsv", sep="\t", header=None)
print(f"test3 {test3.shape}")
test3[2] = -1
test3["T"] = "test3"
# L.append(test3)

test5 = pd.read_csv("../xfdata/5/test.csv", sep="\t", header=None)
print(f"test5 {test5.shape}")
test5[2] = -1
test5["T"] = "test5"
# L.append(test5)

for task in [
    "bq_corpus",
    "lcqmc",
    "paws-x-zh",
]:
    test6 = pd.read_csv(f"../xfdata/6/{task}/test.tsv", sep="\t", header=None)
    print(f"test6 {test6.shape}")
    test6[2] = -1
    test6["T"] = f"test6_{task}"
    L.append(test6)

data = pd.concat(L)
print(pd.value_counts(data["T"]))


#
def get_Embedding(df):
    _X, _Y, _T = [], [], []
    for x1, x2, y, t in tqdm(zip(df[0], df[1], df[2], df["T"]), total=len(df)):
        _X.append([x1, x2])
        _Y.append(y)
        _T.append(t)

    data = []
    for _x in tqdm(_X):
        hl = hashlib.md5()
        hl.update(f"{_x[0]}\t{_x[1]}".encode(encoding='utf-8'))
        _temp = f"/data/temp/{hl.hexdigest()}"

        if not os.path.exists(_temp):
            try:
                Embedding = erniebot.Embedding.create(
                    model='ernie-text-embedding',
                    input=_x
                ).get_result()
                time.sleep(1)

                EMB = np.concatenate(Embedding, axis=0)
                with open(_temp, "wb") as f:
                    pickle.dump(EMB, f)
            except:
                print("Too Long...")
        else:
            with open(_temp, "rb") as f:
                EMB = pickle.load(f)
        #
        data.append(EMB)

    # data = pd.DataFrame(data)
    # data["Y"] = _Y
    # data["T"] = _T

    #
    N = 768
    for _type in [
        "trainE1",
        "trainE2",
        # "test3",
        # "test5",
        "test6_bq_corpus", 
        "test6_lcqmc", 
        "test6_paws-x-zh",
    ]:
        # data[data["T"] == _type].to_csv(f"../user_data/cut_data/{_type}_EMB.csv", index=False)
        with open(f"../user_data/cut_data/{_type}_EMB.csv", "w") as f:
            f.write(f'{",".join([str(i) for i in range(N)])},Y,T\n')
            for _d, _y, _t in tqdm(
                zip(data, _Y, _T), 
                desc=_type,
                total=len(data)
            ):
                if _type == _t:
                    f.write(f'{",".join([str(_d[i]) for i in range(N)])},{_y},{_t}\n')


get_Embedding(df=data)


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
trainE1 = pd.read_csv("../user_data/cut_data/trainE1.csv", sep="\t", header=None)
trainE2 = pd.read_csv("../user_data/cut_data/trainE2.csv", sep="\t", header=None)
print(trainE1.shape, trainE2.shape)

_, trainE1 = train_test_split(trainE1, test_size=5)
_, trainE2 = train_test_split(trainE2, test_size=5)
print(trainE1.shape, trainE2.shape)


def get_Embedding(df, name):
    _X, _Y = [], []
    for x1, x2, y in tqdm(zip(df[0], df[1], df[2]), total=len(df)):
        response = erniebot.Embedding.create(
            model='ernie-text-embedding',
            input=[x1, x2])

        _X.append(np.concatenate(response.get_result(), axis=0))
        _Y.append(y)
        break

    data = pd.DataFrame(_X)
    data["Y"] = _Y
    data.to_csv(f"../user_data/cut_data/{name}_EMB.csv")


# get_Embedding(df=trainE1, name="trainE1")
get_Embedding(df=trainE2, name="trainE2")


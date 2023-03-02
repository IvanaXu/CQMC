import pandas as pd
from sklearn.model_selection import train_test_split
from paddlenlp.datasets import load_dataset
from sklearn.utils import shuffle

# wrongE1 = pd.read_csv("../user_data/model_data/model_state.pdparams-result-trainE1.csv")
# wrongE2 = pd.read_csv("../user_data/model_data/model_state.pdparams-result-trainE2.csv")
# print(wrongE1, wrongE2)

#
train = pd.read_csv("../xfdata/train.csv", sep="\t", header=None)
print("E", train.shape)

trainE1, trainE2 = train_test_split(
    train,
    test_size=100,
    random_state=930721,
)
trainE1 = pd.concat([
    trainE1,
    # wrongE1[["0", "1", "label"]].rename({"0": 0, "1": 1, "label": 2}, axis=1),
    # wrongE2[["0", "1", "label"]].rename({"0": 0, "1": 1, "label": 2}, axis=1),
])
trainE1 = shuffle(trainE1, random_state=10086)

print("E1 E2", trainE1.shape, trainE2.shape)
trainE1.to_csv("../user_data/cut_data/trainE1.csv", index=False, header=False, sep="\t")
trainE2.to_csv("../user_data/cut_data/trainE2.csv", index=False, header=False, sep="\t")


def read(data_path):
    with open(data_path, 'r') as f:
        for line in f:
            query, title, label = line.strip('\n').split('\t')
            yield {'query': query, 'title': title, 'label': int(label)}


for i in load_dataset(read, data_path='../user_data/cut_data/trainE1.csv', lazy=False):
    print(i)
    break

for i in load_dataset(read, data_path='../user_data/cut_data/trainE2.csv', lazy=False):
    print(i)
    break

#
import os
import time
import numpy as np
import pandas as pd
import paddle as pdl
from tqdm import tqdm
from paddle import optimizer
from sklearn.metrics import f1_score

NW = 0
BATCH = 256
NNN = 384
X_cols, Y_cols = [str(i) for i in range(NNN*2)], ["Y"]

trainE1 = pd.read_csv("../user_data/cut_data/trainE1_EMB.csv")
print(pd.value_counts(trainE1["Y"]))
trainE2 = pd.read_csv("../user_data/cut_data/trainE2_EMB.csv")
print(pd.value_counts(trainE2["Y"]))
print(trainE1.shape, trainE2.shape)

trainE1_X, trainE1_Y = trainE1[X_cols].values, trainE1[Y_cols].values
trainE2_X, trainE2_Y = trainE2[X_cols].values, trainE2[Y_cols].values


class Dataset(pdl.io.Dataset):
    def __init__(self, _n, _x, _y):
        self._n = _n
        self._x = _x
        self._y = _y

    def __getitem__(self, index):
        x, y = self._x[index], self._y[index]
        return x.astype(np.float32), y.astype(np.int32)

    def __len__(self):
        return len(self._x)


# 读取 data_loaders
trainE1_loaders = pdl.io.DataLoader(
    Dataset("trainE1", trainE1_X, trainE1_Y),
    return_list=True, shuffle=True, batch_size=BATCH, drop_last=True,
    num_workers=NW,
)
trainE2_loaders = pdl.io.DataLoader(
    Dataset("trainE2", trainE2_X, trainE2_Y),
    return_list=True, shuffle=True, batch_size=BATCH, drop_last=True,
    num_workers=NW,
)


# 读取 model
class PaiPai(pdl.nn.Layer):
    def __init__(self):
        super(PaiPai, self).__init__()
        self.model = pdl.nn.Sequential(
            pdl.nn.Linear(in_features=NNN, out_features=256),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=256, out_features=128),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=128, out_features=64),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=64, out_features=32),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=32, out_features=16),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=16, out_features=8),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=8, out_features=4),
            pdl.nn.ReLU(),
            pdl.nn.Dropout(0.5),

            pdl.nn.Linear(in_features=4, out_features=2),
        )
        self.b0 = self.create_parameter(
            [NNN], is_bias=True, default_initializer=pdl.nn.initializer.Constant(value=0.0))
        self.w1 = self.create_parameter(
            [NNN], is_bias=True, default_initializer=pdl.nn.initializer.Constant(value=1.0))
        self.w2 = self.create_parameter(
            [NNN], is_bias=True, default_initializer=pdl.nn.initializer.Constant(value=1.0))

    def forward(self, _x):
        _x1, _x2 = _x[:, :NNN], _x[:, NNN:]
        return self.model(self.w1 * _x1 + self.w2 * _x2 + self.b0)


def get_feature(_encoder, _data_loader, _tqdm="", batch_size=BATCH):
    _X, _Y = [], []
    data_loader = _data_loader if not _tqdm else tqdm(_data_loader, desc=_tqdm)
    for (_x, _label) in data_loader:
        _r = _encoder(_x)

        _X.append(_r.cpu().numpy())
        _Y.append(_label)
    return np.concatenate(_X, axis=0), np.concatenate(_Y, axis=0)


def fscore(y, _y):
    y, _y = y[:, 0], np.argmax(_y, axis=1)
    return f1_score(y_true=y, y_pred=_y, average="binary")


encoder = PaiPai()

# 损失函数
criterion = pdl.nn.loss.MSELoss()
# 余弦退火学习率 learning_rate=1e-3
scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=0.01, T_max=10)
# 优化器Adam
opt = optimizer.Adam(
    scheduler,
    parameters=encoder.parameters(),
    weight_decay=1e-5,
)


mdl = "/Volumes/ESSD/TEMP/model/"
# os.system(f"rm -rf {mdl}/*")

opt_pkl, encoder_pkl = f"{mdl}/model.opt", f"{mdl}/model.mdl"
if not os.path.exists(f"{mdl}/model.mdl"):
    if os.path.exists(f"{mdl}/BEST.model.mdl"):
        print("> Load BEST.model.mdl.")
        encoder.set_state_dict(pdl.load(f"{mdl}/BEST.model.mdl"))
    if os.path.exists(f"{mdl}/BEST.model.opt"):
        print("> Load BEST.model.opt.")
        opt.set_state_dict(pdl.load(f"{mdl}/BEST.model.opt"))

    # Paras
    NTASK, NSTOP = 99999999, 500
    start = time.perf_counter()
    current_best_metric = -np.inf
    max_bearable_epoch = NSTOP  # 设置早停的轮数为50，若连续50轮内验证集的评价指标没有提升，则停止训练
    current_best_epoch = 0

    # Train 开始
    ST = 1
    for epoch in range(ST, NTASK + 1):
        if epoch > 1:
            encoder.train()
            for (_x, _label) in trainE1_loaders:
                _Yt = encoder(_x)
                Yt = pdl.to_tensor(_label, dtype=pdl.float32, place=pdl.CPUPlace())
                #

                loss = criterion(_Yt, Yt)
                loss.backward()  # 反向传播
                opt.step()  # 更新参数
                opt.clear_grad()
            scheduler.step()  # 更新学习率

        # Valid
        encoder.eval()
        # print(
        #     f"\n"
        #     f"b0: {encoder.b0.cpu().numpy().mean():.6f}, "
        #     f"w1: {encoder.w1.cpu().numpy().mean():.6f}, "
        #     f"w2: {encoder.w2.cpu().numpy().mean():.6f}, "
        # )
        train_X, train_Y = get_feature(_encoder=encoder, _data_loader=trainE1_loaders, _tqdm="")
        valid_X, valid_Y = get_feature(_encoder=encoder, _data_loader=trainE2_loaders, _tqdm="")

        score_train = fscore(train_Y, train_X)
        score_valid = fscore(valid_Y, valid_X)
        _score = score_valid
        score = _score

        if score > current_best_metric:
            scoreR = f"{mdl}/s{_score:.4f}_t{score_train:.4f}_v{score_valid:.4f}"
            # 保存score最大时的模型权重
            current_best_metric = score
            current_best_epoch = epoch
            pdl.save(encoder.state_dict(), f"{scoreR}.mdl")
            pdl.save(encoder.state_dict(), encoder_pkl)

            pdl.save(opt.state_dict(), f"{scoreR}.opt")
            pdl.save(opt.state_dict(), opt_pkl)

        print(
            f" |Epoch {epoch / NTASK:7.2%} |Time {(time.perf_counter() - start):10.2f}s"
            f" |Speed {(time.perf_counter() - start) / epoch:6.2f}s/it"
            f" |Now @{epoch:04d} T/{score_train:12.4f} V/{score_valid:12.4f} S/{score:12.4f}"
            f" |Best @{current_best_epoch:03d} {current_best_metric:12.4f} {(epoch - current_best_epoch) / max_bearable_epoch:7.2%}"
            f" {'MAX' if current_best_epoch == epoch else '   '}"
            f" |-{(int(score * 50) * '-') + '>':51s}|",
        )
        if epoch > current_best_epoch + max_bearable_epoch:
            break

#
model = PaiPai()
model.set_state_dict(pdl.load(f"{mdl}/model.mdl"))
model.eval()
BATCH = 1


# > test3
testO = pd.read_csv("../user_data/cut_data/test3_EMB.csv")
print(f"\ntestO/test3 {testO.shape}")
testO_X, testO_Y = testO[X_cols].values, testO[Y_cols].values
testO_loaders = pdl.io.DataLoader(
    Dataset("testO", testO_X, testO_Y),
    return_list=True, shuffle=False, batch_size=BATCH, drop_last=True,
    num_workers=NW,
)

_R = []
with open("../prediction_result/predict.json", "w") as f:
    for (_x, _label) in testO_loaders:
        for _r in model(_x).cpu().numpy():
            _r = np.argmax(_r)
            f.write(f'{{"label": {_r}}}\n')
            _R.append(_r)
print(pd.value_counts(_R))
assert len(_R) == testO.shape[0]


# > test5
testO = pd.read_csv("../user_data/cut_data/test5_EMB.csv")
print(f"\ntestO/test5 {testO.shape}")
testO_X, testO_Y = testO[X_cols].values, testO[Y_cols].values
testO_loaders = pdl.io.DataLoader(
    Dataset("testO", testO_X, testO_Y),
    return_list=True, shuffle=False, batch_size=BATCH, drop_last=True,
    num_workers=NW,
)

_R = []
for (_x, _label) in testO_loaders:
    for _r in model(_x).cpu().numpy():
        _r = np.argmax(_r)
        _R.append(_r)
pd.DataFrame(_R).to_csv("../prediction_result/result5.csv", index=False, header=False)
print(pd.value_counts(_R))
assert len(_R) == testO.shape[0]


# > test6
for task in ["bq_corpus", "lcqmc", "paws-x-zh"]:
    testO = pd.read_csv(f"../user_data/cut_data/test6_{task}_EMB.csv")
    print(f"\ntestO/test6_{task} {testO.shape}")
    testO_X, testO_Y = testO[X_cols].values, testO[Y_cols].values
    testO_loaders = pdl.io.DataLoader(
        Dataset("testO", testO_X, testO_Y),
        return_list=True, shuffle=False, batch_size=BATCH, drop_last=True,
        num_workers=NW,
    )

    _R = []
    for (_x, _label) in testO_loaders:
        for _r in model(_x).cpu().numpy():
            _r = np.argmax(_r)
            _R.append(_r)
    pd.DataFrame(_R, columns=["prediction"]).reset_index()[["index", "prediction"]].to_csv(
        f"../prediction_result/{task.replace('-zh', '')}.tsv", index=False, sep="\t")
    print(task, pd.value_counts(_R))
    assert len(_R) == testO.shape[0]




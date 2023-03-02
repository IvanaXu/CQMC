
### 一、系统依赖
Python 3.7.4，详见requirements.txt
> GPU：NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2


### 二、数据生成
在生成数据部分，初始状态下无wrong部分生成trainE1/trainE2：
```
# code/train/cdata.py
# ...

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

# ...
```
wrongE1/wrongE2部分，由训练后trainE1/trainE2中预测错误部分如：
> 求网页制作成品简单 求简单的网页制作成品 1 但预测为0
>
> 鬼月怪物猎人漫画 怪物猎人漫画鬼月 0 但预测为1

补充进新的训练集，通过加大权重实现数据增强


### 三、核心算法
基于RoBERTa(A Robustly Optimized BERT Pretraining Approach)，支持12层Transformer网络的roberta-wwm-ext的预训练模型，

实验条件下，roberta-wwm-ext ACC 0.87526/0.84904(devp/test)
> FROM https://arxiv.org/abs/1907.11692


### 四、文件结构
```
% tree
├── README.md
├── code
│   ├── log
│   │   ├── endpoints.log
│   │   └── workerlog.0
│   ├── test
│   │   ├── model.py
│   │   └── predict.py
│   ├── test.sh
│   ├── train
│   │   ├── cdata.py
│   │   ├── model.py
│   │   └── train.py
│   └── train.sh
├── prediction_result
│   └── result.csv
├── requirements.txt
├── user_data
│   ├── cut_data
│   │   ├── trainE1.csv
│   │   └── trainE2.csv
│   └── model_data
│       ├── model_state.pdparams
│       ├── model_state.pdparams-result-trainE1.csv
│       ├── model_state.pdparams-result-trainE2.csv
│       ├── model_state.pdparams-result.csv
│       ├── model_state.pdparams-result.json
│       ├── predict.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
└── xfdata
    ├── test.csv
    └── train.csv
```
其中，
* project/user_data/model_data/model_state.pdparams
约400M，未在压缩包内，可通过以下链接下载
> https://ivan-bucket-out-001.oss-cn-beijing.aliyuncs.com/out/model_state.pdparams


### 五、过程复现
#### 1) 训练
```
cd project/code
sh train.sh
```

#### 2) 预测
```
cd project/code
sh test.sh
```


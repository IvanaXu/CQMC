### 一、系统依赖
Python 3.7.4，详见requirements.txt
> GPU：NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2


### 二、数据生成
(2.1)
* 1/LCQMC
* 2/AFQMC
* 3/MOQMC 
* 4/C19QMC 新冠疫情相似问句判定数据集
* 5/iflytek 中文对话文本匹配挑战赛
* 6/千言数据集：文本相似度

| No. Links | Type | *TOP* | Score | UP |
|--|-|-|-|-|
| [TASK-3](http://contest.aicubes.cn/#/detail?topicId=23)                               | F1          | *0.6365* | 0.5865  | - |
| [TASK-5](https://challenge.xfyun.cn/topic/info?type=text-match&option=ssgy)           | ACC         | *0.9998* | 0.9986  | + |
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | ACC         | *0.9428* | 0.8179  | + |
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / lcqmc     | *0.9548* | 0.8474  | + |
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / bq_corpus | *0.9775* | 0.8414  | + |
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / paws-x    | *0.8960* | 0.7650  | + |
| Total | | | 0.8014 | + |
> Updated 2023-03-03.


(2.2)
wrongE1/wrongE2部分，由训练后trainE1/trainE2中预测错误部分如：
> 求网页制作成品简单 求简单的网页制作成品 1 但预测为0
>
> 鬼月怪物猎人漫画 怪物猎人漫画鬼月 0 但预测为1
> 
补充进新的训练集，通过加大权重实现数据增强


### 三、核心算法
> https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching/sentence_transformers
> 

ERNIE(Enhanced Representation through Knowledge Integration)，支持ERNIE 1.0中文模型（简写ernie-1.0）和ERNIE Tiny中文模型（简写ernie-tiny)。 其中ernie由12层Transformer网络组成，ernie-tiny由3层Transformer网络组成。

更多预训练模型，参考:
> https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer
>


### 四、文件结构
```
% tree
├── README.md
├── code
│   ├── log
│   │   ├── endpoints.log
│   │   └── workerlog.0
│   ├── test
│   │   ├── model.py
│   │   └── predict.py
│   ├── test.sh
│   ├── train
│   │   ├── cdata.py
│   │   ├── model.py
│   │   └── train.py
│   └── train.sh
├── prediction_result
│   └── result.csv
├── requirements.txt
├── user_data
│   ├── cut_data
│   │   ├── trainE1.csv
│   │   └── trainE2.csv
│   └── model_data
│       ├── model_state.pdparams
│       ├── model_state.pdparams-result-trainE1.csv
│       ├── model_state.pdparams-result-trainE2.csv
│       ├── model_state.pdparams-result.csv
│       ├── model_state.pdparams-result.json
│       ├── predict.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
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

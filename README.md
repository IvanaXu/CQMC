### 一、系统依赖
Python 3.7.4，详见requirements.txt
> GPU：NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2


### 二、数据生成
(2.1)

> THX
>
> https://github.com/liucongg/NLPDataSet
>
> https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html

* 1/LCQMC
* 2/AFQMC
* 3/MOQMC 
* 4/C19QMC 新冠疫情相似问句判定数据集
* 5/iflytek 中文对话文本匹配挑战赛
* 6/千言数据集：文本相似度
* 7/BUSTM FewCLUE 评测中对话短文本语义匹配数据集 2分类任务
* 8/CMNLI 中文语言推理任务


| No. Links | Type | *TOP* | Score | UP |
|--|-|-|-|-|
| [TASK-3](http://contest.aicubes.cn/#/detail?topicId=23)                               | F1          | *0.6365* | 0.5907 |+|
| [TASK-5](https://challenge.xfyun.cn/topic/info?type=text-match&option=ssgy)           | ACC         | *0.9998* | 0.9954 |+|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | ACC         | *0.9428* | 0.7724 |-|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / lcqmc     | *0.9548* | 0.8360 |-|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / bq_corpus | *0.9775* | 0.8252 |-|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / paws-x    | *0.8960* | 0.6560 |-|
| Total | | | 0.7861 |+|
> Updated 2023-03-07.


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
project
├── code
│   ├── 1.sh
│   ├── 2.sh
│   ├── log
│   │   ├── default.gpu.log
│   │   ├── default.ifyfyf.log
│   │   ├── workerlog.0
│   │   ├── workerlog.1
│   │   ├── workerlog.2
│   │   └── workerlog.3
│   ├── test
│   │   ├── model.py
│   │   ├── predict.py
│   │   └── __pycache__
│   │       └── model.cpython-37.pyc
│   └── train
│       ├── cdata.py
│       ├── model.py
│       ├── __pycache__
│       │   └── model.cpython-37.pyc
│       └── train.py
├── prediction_result
│   ├── bq_corpus.tsv
│   ├── lcqmc.tsv
│   ├── paws-x.tsv
│   ├── predict.json
│   ├── result5.csv
│   └── result6.zip
├── requirements.txt
├── user_data
│   ├── cut_data
│   │   ├── md5
│   │   ├── trainE1.csv
│   │   └── trainE2.csv
│   └── model_data
│       ├── model_state.pdparams
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
└── xfdata
    ├── 1
    │   ├── 2205.11097.pdf
    │   └── similarity_ch.zip
    ├── 2
    │   ├── dev.json
    │   ├── test.json
    │   └── train.json
    ├── 3
    │   ├── test.tsv
    │   └── train.tsv
    ├── 4
    │   ├── test.csv
    │   └── test.label.csv
    ├── 5
    │   ├── test.csv
    │   └── train.csv
    └── 6
        ├── bq_corpus
        │   ├── dev.tsv
        │   ├── License.pdf
        │   ├── test.tsv
        │   ├── train.tsv
        │   └── User_Agreement.pdf
        ├── lcqmc
        │   ├── dev.tsv
        │   ├── License.pdf
        │   ├── test.tsv
        │   ├── train.tsv
        │   └── User_Agreement.pdf
        └── paws-x-zh
            ├── dev.tsv
            ├── License.pdf
            ├── test.tsv
            └── train.tsv
```
其中，未在源码内，可通过以下链接下载：
* project/user_data/model_data/model_state.pdparams
* project/user_data/cut_data/trainE1.csv
* project/user_data/cut_data/trainE2.csv
> 链接: https://pan.baidu.com/s/1gGvyXOtoRtpMGqwTu65aqw 提取码: t3it 


### 五、过程复现
#### 1) 训练
```
cd project/code
sh 1.sh
```

#### 2) 预测
```
cd project/code
sh 2.sh
```

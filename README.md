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

* 1/LCQMC A Large-scale Chinese Question Matching Corpus
* 2/AFQMC 蚂蚁金融语义相似度 Ant Financial Question Matching Corpus
* 3/MOQMC 同花顺算法挑战-跨领域迁移的文本语义匹配
* 4/C19QMC 新冠疫情相似问句判定数据集
* 5/iflytek 中文对话文本匹配挑战赛
* 6/千言数据集：文本相似度
* 7/BUSTM FewCLUE 评测中对话短文本语义匹配数据集 2分类任务
* 8/CMNLI 中文语言推理任务


| No. Links | Type | *TOP* | Score | UP |
|--|-|-|-|-|
| v0.1 SAS                                                                              | F1          | *0.2653* |        |-|
| v0.2 XYSH                                                                             | F1          | *0.3483* |        |-|
| v0.3 CROSS v0.1 + v0.2                                                                | F1          | *0.3696* |        |-|
| [TASK-3](http://contest.aicubes.cn/#/detail?topicId=23)                               | F1          | *0.6365* | 0.5948 |=|
| [TASK-5](https://challenge.xfyun.cn/topic/info?type=text-match&option=ssgy)           | ACC         | *0.9998* | 0.9994 |=|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | ACC         | *0.9428* | 0.7927 |=|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / lcqmc     | *0.9548* | 0.8369 |=|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / bq_corpus | *0.9775* | 0.8337 |=|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / paws-x    | *0.8960* | 0.7075 |=|
| Total | | | 0.9250 |-|
> Updated  2023-03-16 00:04:32.769588, Test ACC: 0.90580 F1: 0.89503.


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

本次开源的模型是文心大模型 ERNIE 3.0, 文心大模型 ERNIE 3.0 作为百亿参数知识增强的大模型，除了从海量文本数据中学习词汇、结构、语义等知识外，还从大规模知识图谱中学习。 基础上通过在线蒸馏技术得到的轻量级模型，模型结构与 ERNIE 2.0 保持一致，相比 ERNIE 2.0 具有更强的中文效果。

相关技术详解可参考文章[《解析全球最大中文单体模型鹏城-百度·文心技术细节》](https://www.jiqizhixin.com/articles/2021-12-08-9)

在线蒸馏技术
在线蒸馏技术在模型学习的过程中周期性地将知识信号传递给若干个学生模型同时训练，从而在蒸馏阶段一次性产出多种尺寸的学生模型。相对传统蒸馏技术，该技术极大节省了因大模型额外蒸馏计算以及多个学生的重复知识传递带来的算力消耗。

这种新颖的蒸馏方式利用了文心大模型的规模优势，在蒸馏完成后保证了学生模型的效果和尺寸丰富性，方便不同性能需求的应用场景使用。此外，由于文心大模型的模型尺寸与学生模型差距巨大，模型蒸馏难度极大甚至容易失效。为此，通过引入了助教模型进行蒸馏的技术，利用助教作为知识传递的桥梁以缩短学生模型和大模型表达空间相距过大的问题，从而促进蒸馏效率的提升。

更多技术细节可以参考论文：

[ERNIE-Tiny: A Progressive Distillation Framework for Pretrained Transformer Compression](https://arxiv.org/abs/2106.02241)
[ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731)

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
│   │   ├── default.dohgxk.log
│   │   ├── default.gpu.log
│   │   └── workerlog.0
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
├── LICENSE
├── ndocker.sh
├── prediction_result
│   ├── bq_corpus.tsv
│   ├── download_20230308223931.zip
│   ├── lcqmc.tsv
│   ├── paws-x.tsv
│   ├── predict.json
│   ├── result5.csv
│   └── result6.zip
├── README.md
├── README.md.py
├── requirements.txt
├── user_data
│   ├── cut_data
│   │   ├── md5
│   │   ├── trainE1.csv
│   │   └── trainE2.csv
│   └── model_data
│       ├── model_state_20230308223931.pdparams
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
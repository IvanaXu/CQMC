
### 写在前面
> https://www.qbitai.com/2023/03/42483.html
```
过去的自然语言专家，擅长于自己的领域，有人专门做文本分类、有人专门做信息抽取、有人做问答、有人做阅读理解。
大家会定义自然语言领域的各种各样的任务，这些任务可能有上百个，非常多。
每个任务都有专门的模型和框架，然后还有专门的专家，根据专门数据训练出来，然后摆在那儿供大家调用，所有这些NLP能力，就像一个工具集，成百上千个工具摆在架子上。
那么这个对于想触达这些能力的人来说，就存在挑战，就是我怎么知道成千上百个工具，哪个是适合我的。
所以还需要算法专家进一步解释，你面临的这个问题是文本分类问题，那个问题是阅读理解问题，再把工具给你。
所以大家可以看到，制造AI能力的人，跟最终使用这个能力的人中间，是巨大的Gap，如何去弥补？
其实我们之前一直没有想到很好的方法，大家做了各种各样的平台，都试图去弥补Gap，但现在看起来都不成功。
最终ChatGPT告诉我们一件事情，弥补AI自然语言能力跟用户之间Gap的方法，就是自然语言本身，让用户他用自然语言去描述，让大模型去理解用户想干什么，然后把这个能力给到它。
举个例子，请描述一下中国足球的未来。
这个容易，如果加一个约束，请简短的用三条来描述，这个在过去的问答系统里边，你就很难让实现，需要算法专家把它专门变成一个有约束的问题。
现在ChatGPT不用了，你能用自然语言去描述你想做什么就可以了，ChatGPT都能理解。
所以大模型实际上缩短了AI能力跟用户之间的距离，所有人都可以用了，一下子就火了。
王宝元博士：那可不可以这么理解，原来很多传统NLP的任务已经不存在了？
张家兴博士：如果我们套用《三体》里面非常著名的一句话，“物理学不存在了”，那么我们今天从某种意义上也可以说，NLP技术不存在了。
王宝元博士：这个讲法非常大胆。
张家兴博士：对，NLP技术不存在了。但还是要加一句解释，只是传统的那种。不再需要单纯的算法专家去设计单个的NLP能力。
那新的NLP方式是什么，就是努力去做一个通用的ChatGPT，把所有提供给用户的能力，都注入到一个模型里，让这个模型可以通过自然语言的方式，给用户提供所有的能力。
```

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
| [TASK-3](http://contest.aicubes.cn/#/detail?topicId=23)                               | F1          | *0.6365* | -1.0000 |-|
| [TASK-5](https://challenge.xfyun.cn/topic/info?type=text-match&option=ssgy)           | ACC         | *0.9998* | -1.0000 |-|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | ACC         | *0.9428* | 0.6739 |+|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / lcqmc     | *0.9548* | 0.7300 |+|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / bq_corpus | *0.9775* | 0.7383 |+|
| [TASK-6](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) | / paws-x    | *0.8960* | 0.5535 |+|
| Total | | | -0.6188 |-|
> Updated  2024-01-10 00:30:59.378136, Test ACC: -1 F1: 0.6863, By [erniebot](./erniebot).


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

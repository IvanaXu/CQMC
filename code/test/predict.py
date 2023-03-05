# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad

from model import SentenceTransformer

from tqdm import tqdm
import pandas as pd
from sklearn import metrics

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default='', help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=50, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--cN", default=None, type=int, help="cN for test to predict the model result")
args = parser.parse_args()
# yapf: enable


def convert_example(example, tokenizer, max_seq_length=512):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed
    to be used in a sequence-pair classification task.

    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing query, title and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.

    Returns:
        query_input_ids(obj:`list[int]`): The list of query token ids.
        query_token_type_ids(obj: `list[int]`): List of query sequence pair mask.
        title_input_ids(obj:`list[int]`): The list of title token ids.
        title_token_type_ids(obj: `list[int]`): List of title sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    query, title = example[0], example[1]

    query_encoded_inputs = tokenizer(text=query, max_seq_len=max_seq_length)
    query_input_ids = query_encoded_inputs["input_ids"]
    query_token_type_ids = query_encoded_inputs["token_type_ids"]

    title_encoded_inputs = tokenizer(text=title, max_seq_len=max_seq_length)
    title_input_ids = title_encoded_inputs["input_ids"]
    title_token_type_ids = title_encoded_inputs["token_type_ids"]

    return query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids


def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text_pair in tqdm(data, desc="Step 1"):
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = convert_example(
            text_pair, tokenizer, max_seq_length=args.max_seq_length)
        examples.append((query_input_ids, query_token_type_ids, title_input_ids,
                         title_token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in tqdm(range(0, len(examples), batch_size), desc="Step 2")
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tilte_segment
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in tqdm(batches, desc="Step 3"):
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batchify_fn(
            batch)

        query_input_ids = paddle.to_tensor(query_input_ids)
        query_token_type_ids = paddle.to_tensor(query_token_type_ids)
        title_input_ids = paddle.to_tensor(title_input_ids)
        title_token_type_ids = paddle.to_tensor(title_token_type_ids)

        probs = model(query_input_ids,
                      title_input_ids,
                      query_token_type_ids=query_token_type_ids,
                      title_token_type_ids=title_token_type_ids)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    # ErnieTinyTokenizer is special for ernie-tiny pretained model.
    # ernie-tiny/roberta-wwm-ext
    pretrained_model = AutoModel.from_pretrained('ernie-3.0-nano-zh')
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-nano-zh')
    model = SentenceTransformer(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)


    #
    def get_predict(data, label="", name=""):
        label_map = {0: 'dissimilar', 1: 'similar'}
        result_map = {_v: _k for _k, _v in label_map.items()}

        r_predict = predict(model, data, tokenizer, label_map, batch_size=args.batch_size)
        result = [result_map.get(r_predict[idx]) for idx, text in enumerate(data)]

        if len(label) > 0:
            print(f"\n{name} Accuracy_score: {metrics.accuracy_score(label, result):.5f} F1: {metrics.f1_score(label, result):.5f}")

            _data = pd.DataFrame(data)
            _data[0] = _data[0].apply(lambda x: f"|{x}|")
            _data[1] = _data[1].apply(lambda x: f"|{x}|")
            _data["label"], _data["result"] = label, result
            _data = _data[_data["label"] != _data["result"]]
            
            # print(_data.to_string())
            # _data.to_csv(f"{args.params_path}-result-{name}.csv", index=False)
        else:
            return result

    # Predict Data
    # """
    # trainE1 = pd.read_csv("../user_data/cut_data/trainE1.csv", sep="\t", header=None, nrows=args.cN)
    trainE2 = pd.read_csv("../user_data/cut_data/trainE2.csv", sep="\t", header=None, nrows=args.cN)

    # get_predict(trainE1[[0, 1]].to_numpy(), trainE1[2], "trainE1")
    get_predict(trainE2[[0, 1]].to_numpy(), trainE2[2], "trainE2")
    # """
    
    
    # 3
    test3 = pd.read_csv("../xfdata/3/test.tsv", sep="\t", header=None, nrows=args.cN)
    test3 = get_predict(test3[[0, 1]].to_numpy())
    with open("../prediction_result/predict.json", "w") as f:
        for _r in test3:
            f.write(f'{{"label": {_r}}}\n')

    # 5
    test5 = pd.read_csv("../xfdata/5/test.csv", sep="\t", header=None, nrows=args.cN)
    test5 = get_predict(test5[[0, 1]].to_numpy())
    pd.DataFrame(test5).to_csv("../prediction_result/result5.csv", index=False, header=False)

    # 6
    for task in ["bq_corpus", "lcqmc", "paws-x-zh"]:
        test6 = pd.read_csv(f"../xfdata/6/{task}/test.tsv", sep="\t", header=None, nrows=args.cN)
        test6 = get_predict(test6[[0, 1]].to_numpy())
        pd.DataFrame(test6, columns=["prediction"]).reset_index()[["index", "prediction"]].to_csv(
            f"../prediction_result/{task.replace('-zh','')}.tsv", index=False, sep="\t")
    # Predict Data




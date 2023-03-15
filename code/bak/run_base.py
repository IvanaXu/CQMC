# -*--coding:utf-8*-
# @auth ivan
# @time 20210613 14:50:00
# @goal test .

import json
import pandas as pd
from tqdm import tqdm

datap = "/Users/ivan/Desktop/ALL/Data/AddressNLP3"
datan = "Xeon3NLP_round1"
# "Xeon3NLP_round1_train_20210524.txt"
# "Xeon3NLP_round1_test_20210524.txt"

r_result = []
with open(f"{datap}/{datan}_train_20210524.txt") as f:
    for i in tqdm(f):
        j = json.loads(i)
        
        for _j in j["candidate"]:
            r_result.append([j["text_id"], j["query"], _j["text"], "Trai", _j["label"]])
        
with open(f"{datap}/{datan}_test_20210524.txt") as f:
    for i in tqdm(f):
        j = json.loads(i)
        
        for _j in j["candidate"]:
            r_result.append([j["text_id"], j["query"], _j["text"], "Test", ""])
        
r_result = pd.DataFrame(
    r_result, 
    columns=["text_id", "query", "text", "types", "label"]
)
print(r_result)
r_result.to_csv(
    f"{datap}/{datan}_SAS_in.txt", 
    index=False, sep=chr(1), encoding="utf-8"
)

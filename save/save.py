
import os
import time
from tqdm import tqdm

for _ in tqdm(range(9999)):
    os.system("sh save.sh")
    time.sleep(60 * 10)

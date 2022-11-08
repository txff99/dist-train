
import os,sys
import torch
import random
import numpy as np

from transformers import BertTokenizer

from torch.utils.data import Dataset

if len(sys.argv)!=3:
    print("check in.txt out.txt")
    sys.exit()

tokenizer = BertTokenizer.from_pretrained("../chinese-macbert-base/")

fin = open(sys.argv[1],"r")
fout = open(sys.argv[2],"w")

while True:
    line = fin.readline()
    if len(line)==0:
        break
    line = line.strip()
    try:
        tokens = tokenizer.tokenize(line)
        ids = tokenizer.convert_tokens_to_ids(tokens)
    except:
        print(line)
        continue

    fout.write(line+"\n")

fin.close()
fout.close()



import os,sys
import torch
import random
import numpy as np
import time
import random
import tqdm
import gc
import copy

from transformers import BertTokenizer

from torch.utils.data import Dataset

class LMTextDataset(Dataset):
    def __init__(self, pre_train_dir,txt_file):
        self.pad_index = 0
        self.unk_index = 100
        self.cls_index = 101
        self.sep_index = 102
        self.mask_index = 103
        self.vocab_size = 21128

        self.tokenizer = BertTokenizer.from_pretrained(pre_train_dir)

        self.examples = []
        self.targets = []

        ftxt = open(txt_file,"r")
        lines = ftxt.readlines()
        ftxt.close()

        for line in lines:
            line = line.strip()
            tokens = self.tokenizer.tokenize(line)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            #self.targets.append(np.array(ids))
            #masked_ids = self.random_sent_ids(ids)
            #self.examples.append(masked_ids)

            target = copy.deepcopy(ids)
            masked_pos,masked_ids,mask,target_id = self.random_sent_ids(ids)
            #print(mask)
            #sys.exit()
            self.examples.append(np.array(masked_ids))
            self.targets.append(np.array(target))


    def random_sent_ids(self,sent_ids):
        mask_pos = random.randint(0,len(sent_ids)-1)
        target_pos_id = sent_ids[mask_pos]
 
        sent_ids[mask_pos] = self.mask_index

        mask = []
        for i in range(0,len(sent_ids)):
            if i==mask_pos:
                mask.append(True)
            else:
                mask.append(False)

        return mask_pos,sent_ids,mask,target_pos_id
        #output_label = []
        #for i,ids in enumerate(sent_ids):
        #    prob = random.random()
        #    if prob < 0.15:
        #        prob /= 0.15
        #        if prob < 0.8:
        #            sent_ids[i] = self.mask_index
        #        elif prob < 0.9:
        #            sent_ids[i] = random.randint(0,self.vocab_size-1)
        #        else:
        #            sent_ids[i] = self.unk_index

        #    output_label.append(sent_ids[i])

        #return np.array(output_label)

    def __getitem__(self,index):
        return ( torch.from_numpy(self.examples[index]),torch.from_numpy(self.targets[index]) )

    def __len__(self):
        return len(self.examples)


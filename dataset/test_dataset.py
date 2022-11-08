
import sys
#sys.path.append("..")

from torch.utils.data import DataLoader
from bert_lm_dataset import LMTextDataset
from cn_collate_func import simple_chn_collate_func

txt_dataset = LMTextDataset("/data/text_data/roberta-chinese/char_model/chinese_roberta_L-4_H-128",
        "/data/text_data/test.txt")

test_iter = DataLoader(txt_dataset,batch_size=4,num_workers=1,collate_fn=simple_chn_collate_func,shuffle=True)
index = 0
for test_item in test_iter:
    print(index)
    print(test_item)
    index += 1
    if index>5:
        sys.exit()




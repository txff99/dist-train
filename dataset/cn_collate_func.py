import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def simple_chn_collate_func(batch_from_dataloader):
    lab_batch = []
    target_batch = []
    for data in batch_from_dataloader:
        lab_batch.append(data[0])
        target_batch.append(data[1])

    #print(lab_batch)
    lab_batch = pad_sequence(lab_batch,batch_first=True)
    target_batch = pad_sequence(target_batch,batch_first=True)
    #target_batch = torch.from_numpy(np.array(target_batch))

    return {'input_label': lab_batch, 'target_label': target_batch}


import numpy as np
import torch
from torch.optim import Adam
from optim_schedule import ScheduledOptim

class FakeOptim():
    def __init__(self,param_group):
        self.param_groups = []
        self.param_groups.append(param_group)

#optim = Adam(param,lr=1e-4,betas=(0.9,0.999),weight_decay=0.01)
param_group = {'lr': 1e-4}
optim = FakeOptim(param_group)

sch = ScheduledOptim(optim,256,n_warmup_steps=300000)
for i in range(0,1000000):
    sch._update_learning_rate()
    if i % 100000==0:
        print(sch.lr)


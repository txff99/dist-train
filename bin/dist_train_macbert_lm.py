
import argparse
import sys
import yaml
import logging
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import BertModel,BertForMaskedLM
from tensorboardX import SummaryWriter

sys.path.append("..")
from model.language_model import BERTLM
from model.bert import BERT
from trainer.dist_macbert_trainer import DistBERTTrainer
from dataset.bert_lm_dataset import LMTextDataset
from dataset.vocab import WordVocab
from dataset.cn_collate_func import simple_chn_collate_func
from utils.train_utils import *

def load_configs(args):
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # save args parameters
    for arg in vars(args):
        configs[arg] = getattr(args, arg)

    configs['model_store_path'] = configs['output_dir']+"/model/"
    os.makedirs(configs['model_store_path'], exist_ok=True)
    if not os.path.exists(configs['model_store_path']):
        print("Can not create ",configs['model_store_path'])
        sys.exit()

    configs['local_rank'] = int(os.environ['LOCAL_RANK'])

    stream_handle = logging.StreamHandler()
    stream_handle.setLevel(logging.INFO)
    if configs['local_rank'] == 0:
        file_handle = logging.FileHandler(configs['train_log_file'])
        file_handle.setLevel(logging.WARNING)
        handlers = [file_handle, stream_handle]
    else:
        handlers = [stream_handle]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s(line:%(lineno)d) [%(levelname)s] %(message)s',
                        handlers=handlers)

    return configs

def parse_ckpt_path(ckpt_path):
    phelm = ckpt_path.split(".")
    elm = phelm[1].split("_")
    epoch = int(elm[0].replace("ep",""))
    step = int(elm[1])
    return epoch,step

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument("--config", type=str, default=None, help="training config files")
    parser.add_argument("--check_point", type=str, default=None, help="check_point of training model")

    args = parser.parse_args()

    configs = load_configs(args)

    if configs['local_rank'] == 0:
        if not os.path.exists(configs['tensorboard_dir']):
            os.makedirs(configs['tensorboard_dir'], exist_ok=True)
        writer = SummaryWriter(configs['tensorboard_dir']) #create_summary_writer(configs)
        collect_environment_info(True)
        print("Loading Vocab", configs['vocab_file'])
    else:
        writer = None


    fdict = open(configs['vocab_file'],"r")
    lines = fdict.readlines()
    fdict.close()
    vocab = WordVocab(lines[4:])
    print("Vocab Size: ", len(vocab))

    fscp = open(configs['train_txt_scp'],"r")
    train_files = fscp.readlines()
    fscp.close()

    random.shuffle(train_files)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(configs['local_rank'])

    print("Loading Test Dataset", configs['test_txt'])
    test_dataset = LMTextDataset(configs['pretrain_model'], configs['test_txt'])

    print("Creating Test Dataloader")

    test_data_loader = DataLoader(test_dataset,batch_size=configs['batch_size'],
            num_workers=configs['num_workers'],
            collate_fn=simple_chn_collate_func,
            shuffle=False)

    start_epoch = 0
    start_step = 0

    if configs['check_point'] == None: 
        print("Building BERT model")
        bert = BERTLM()
        # bert = BertModel.from_pretrained(configs['pretrain_model'])
        # bert = BertForMaskedLM.from_pretrained(configs['pretrain_model'])

        print("Creating BERT Trainer")
        trainer = DistBERTTrainer(bert, len(vocab), test_dataloader=test_data_loader,
                              lr=configs['lr'], betas=(configs['adam_beta1'], configs['adam_beta2']), 
                              weight_decay=configs['adam_weight_decay'],
                              warmup_steps=300000,writer=writer,
                              local_rank=configs['local_rank'], log_freq=configs['log_freq'])
    else:
        model = torch.load(configs['check_point'])
        # print(model)
        start_epoch,start_step = parse_ckpt_path(configs['check_point'])
        trainer = DistBERTTrainer(None, len(vocab), test_dataloader=test_data_loader, 
                              model = model, start_step = start_step,
                              lr=configs['lr'], betas=(configs['adam_beta1'], configs['adam_beta2']),
                              weight_decay=configs['adam_weight_decay'],
                              warmup_steps=300000,writer=writer,
                              local_rank=configs['local_rank'], log_freq=configs['log_freq'])

    if configs['local_rank'] == 0:
        total_params = sum(p.numel() for p in trainer.model.parameters())
        logging.warning("{} {} {}".format("="*20, "MacBert Model", "="*20))
        logging.warning("Parameters size: {}".format(total_params))
        logging.warning(trainer.model)

    print("Training Start")
    for epoch in range(start_epoch, configs['epochs']):
        block_number = len(train_files)
        block_index = 0
        for train_file in train_files:
            train_file = train_file.strip()
            block_index += 1
            print("Training with ",train_file,block_index,"/",block_number)

            if configs['local_rank'] not in [0,-1]:
                torch.distributed.barrier()

            train_dataset = LMTextDataset(configs['pretrain_model'], train_file)

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

            train_data_loader = DataLoader(train_dataset,batch_size=configs['batch_size'],
                    pin_memory=True,
                    num_workers=configs['num_workers'],
                    collate_fn=simple_chn_collate_func,
                    sampler=train_sampler)
                    #shuffle=True)
            
            if configs['local_rank']==0:
                torch.distributed.barrier()

            trainer.load_train_data(train_data_loader)

            trainer.train(epoch)

            torch.distributed.barrier()

            if ( block_index % int(configs['save_per_blocks']) )==0:
                if configs['local_rank']==0:
                    trainer.save(epoch, configs['model_store_path'])

                if test_data_loader is not None:
                    trainer.test(epoch)
            

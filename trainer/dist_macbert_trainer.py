
import sys
import torch
import logging
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import BertModel
from model.language_model import BERTLM
from trainer.optim_schedule import ScheduledOptim
from utils.checkpoint import *
from transformers import BertModel,BertForMaskedLM

import tqdm


class DistBERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    #def __init__(self, bert: BERT, vocab_size: int,
    def __init__(self, bert: None, vocab_size: int,
                 test_dataloader: DataLoader = None, model = None, start_step = 0,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 local_rank=0, log_freq: int = 10, writer=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param local_rank: traning with cuda device rank
        :param log_freq: logging frequency of the batch iteration
        """
        #self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.local_rank = local_rank
        self.device = torch.device(local_rank)

        if model==None:
            # This BERT model will be saved every epoch
            #self.hidden = len(bert.pooler.dense.bias)
            #self.hidden = len(bert.encoder.last_hidden_state)
            self.hidden = bert.encoder.layer[0].output.dense.out_features
            self.bert = bert
            # Initialize the BERT Language Model, with BERT model
            self.model = BERTLM().to(self.device)
            # self.model = BertForMaskedLM.from_pretrained("/data/text_data/roberta-chinese/char_model/chinese_roberta_L-4_H-128")#BERTLM().to(self.device)
            
            self.steps = 0
        else:
            self.model = model.to(self.device)
            #self.hidden = len(self.model.bert.pooler.dense.bias)
            self.hidden = self.model.encoder.layer[0].output.dense.out_features
            self.steps = start_step
        # Distributed GPU training if CUDA can detect more than 1 GPU
        print("Using %d - %d GPUS for BERT" % (torch.cuda.device_count(),local_rank))
        #self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        if torch.cuda.device_count() > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model,device_ids=[local_rank],
                find_unused_parameters=True)

        # Setting the train and test data loader
        self.train_data = None
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        self.optim_schedule = ScheduledOptim(self.optim, self.hidden, n_warmup_steps=warmup_steps)
        self.lr = self.optim_schedule.lr

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion = nn.CrossEntropyLoss()

        self.log_freq = log_freq

        self.writer = writer

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        print("[LOGGING]", local_rank, "Writer: ",self.writer)

    def load_train_data(self,train_dataloader: DataLoader):
        self.train_data = train_dataloader

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        #data_iter = tqdm.tqdm(enumerate(data_loader),
        #                      desc="EP_%s:%d" % (str_code, epoch),
        #                      total=len(data_loader),
        #                      bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        #for i, data in data_iter:
        i = 0

        log_info = "Rank_%d batch_size:%d" % (self.local_rank,len(data_loader))
        print(log_info)

        for data in data_loader:
            # 0. batch_data will be sent into the device(GPU or cpu)
            #data = {key: value.to(self.device) for key, value in data.items()}
            input_label = data['input_label'].to(self.device)
            target_label = data['target_label'].to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm model
            #next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            mask_lm_output = self.model.forward(input_label)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            #next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            # print(mask_lm_output.shape)
            # print(target_label.shape)
            #sys.exit()
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), target_label)
            # mask_loss = self.criterion(mask_lm_output, target_label)
            # print(mask_loss)
            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            #loss = next_loss + mask_loss
            loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            self.lr = self.optim_schedule.lr
            # next sentence prediction accuracy
            #correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            #total_correct += correct
            #total_element += data["is_next"].nelement()

                #"avg_acc": total_correct / total_element * 100,
            self.steps += 1
            i += 1

            if i % self.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                    "lr": self.lr
                }
                #data_iter.write(str(post_fix))
                #if self.writer==None:
                #    print("Can't write tersorboard graph, writer not initial.")
                if train==True and self.writer!=None:
                    log_info = "Writer - train_loss: %.10f, step: %d" % (loss.item(),self.steps)
                    print(log_info)
                    #logging.warning(log_info)
                    self.writer.add_scalar('train/loss',loss.item(),self.steps)
                    self.writer.add_scalar('train/lr',self.lr,self.steps)
                if train==False and self.writer!=None:
                    self.writer.add_scalar('cv/loss',loss.item(),self.steps)

        #log_info = "EP%d_%s, avg_loss=%.03f, lr=%.10f" % (epoch, str_code, avg_loss / len(data_iter), self.lr)
        log_info = "Rank_%d - EP%d_%d_%s, avg_loss=%.03f, lr=%.10f" % \
            (self.local_rank, epoch, self.steps, str_code, avg_loss / len(data_loader), self.lr)
        print(log_info)
        logging.warning(log_info)

    def save(self, epoch, file_path="output/"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if self.local_rank==0:
            output_path = file_path + "/macbert_model.ep%d_%d.pt" % (epoch,self.steps)
            #torch.save(self.model, output_path)
            #model = self.model
            #torch.save(model.cpu(), output_path)
            torch.save(self.model.module, output_path)
            #save_checkpoint(
            #    model=self.model,
            #    filename=output_path,
            #    optimizer=self.optim,
            #    infos=None
            #)

            print("EP:%d-%d Model Saved on:" % (epoch,self.steps), output_path)
            return output_path

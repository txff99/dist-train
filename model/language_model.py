import torch.nn as nn

#from .bert import BERT
from transformers import BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    #def __init__(self, bert: BERT, vocab_size):
    #def __init__(self, bert: BertModel, vocab_size):
    def __init__(self,vocab_size=21128):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        #self.bert = bert
        #self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        # self.embeddings = embedding
        # self.encoder = encoder

        # #bert_hidden = len(bert.encoder.last_hidden_state)
        # hidden_size = encoder.layer[0].output.dense.out_features

        # self.mask_lm = MaskedLanguageModel(hidden_size, vocab_size)
        MODEL_PATH = '/data/text_data/roberta-chinese/char_model/chinese_roberta_L-4_H-128'#'D:\language_model\lib\chinese_roberta_L-4_H-128'
        mask_lm= AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
        # mask_lm= AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
        
        self.embeddings = mask_lm.bert.embeddings
        self.encoder = mask_lm.bert.encoder
       
        self.cls = mask_lm.cls



    def forward(self,x):
    #def forward(self, x, segment_label):
    #    x = self.bert(x, segment_label)
        y = self.embeddings(x)
        y = self.encoder(y)
        encoder_out = y.last_hidden_state
        #print("encoder_out:",encoder_out,encoder_out.shape)
        return self.cls(encoder_out)
        #return self.mask_lm(y)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

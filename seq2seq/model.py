import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_p = dropout_p
        self.gru = nn.GRU(input_size, hidden_size, dropout=dropout_p)
    def forward(self, input, hidden):
        #input.double()
        #hidden.double()
        output, hidden = self.gru(input, hidden)
        return output, hidden
class BahadanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahadanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
    def forward(self, query, keys):
        #get cosine similarity between query and keys
        cos_scores = []
        for i in range(len(keys)):
            vct1 = query.squeeze()
            vct2 = keys[i].squeeze()
            cos = torch.dot(vct1, vct2) / (torch.norm(vct1) * torch.norm(vct2))
            cos_scores.append(cos)
        cos_scores = torch.tensor(cos_scores)
        # scale keys by respective cos scores
        scaled_keys = []
        for i in range(len(keys)):
            scaled_keys.append(cos_scores[i] * keys[i])
        scaled_keys = torch.stack(scaled_keys)
        # get context vector
        context = torch.sum(scaled_keys, dim=0)
        context = context.unsqueeze(0)
        return context
class Attention_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(Attention_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.nax_len = 30
        self.attention = BahadanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, encoder_output, encoder_hidden, out_len):
        decoder_hidden = encoder_hidden
        decoder_input = torch.zeros(1, 1, self.hidden_size, device=device)
        context = torch.zeros(1, 1, self.hidden_size, device=device)
        outputs = []
        for i in range(out_len):
            context = self.attention(decoder_hidden, encoder_output)
            decoder_input = torch.cat((context, decoder_input), dim=2)
            #print(decoder_input.shape, decoder_hidden.shape)
            output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            decoder_input = output
            output = self.out(output)
            output = self.dropout(output)
            outputs.append(output)
        return output
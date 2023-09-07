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
import csv
import re
from dataset import Dataset
from model import EncoderRNN, Attention_Decoder
    
file1 = 'file1.csv'
samples = []
labels = []
tot_count = 0
ques_count = 0
def preprocess(corp_str):
    #replace urls with urlhere
    url_reg  = r'[a-z]*[:.]+\S+'
    corp_str = re.sub(url_reg, 'urlhere', corp_str)
    # replace multiple spaces with a single space
    spaces_reg = r'\s+'
    corp_str = re.sub(spaces_reg, ' ', corp_str)
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                    '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    corp_str = re.sub(url_regex, 'URLHERE', corp_str)
    corp_str = re.sub(mention_regex, 'MENTIONHERE', corp_str)
    corp_str = re.sub(hashtag_regex, 'HASHHERE', corp_str)
    #replac nums with numhere
    num_reg = r'[0-9]+'
    corp_str = re.sub(num_reg, 'numhere', corp_str)
    # replace all punctuations with empty string
    punct_reg = r'[^a-zA-Z0-9\s]'
    corp_str = re.sub(punct_reg, '', corp_str)
    return corp_str.split()
i = 0
with open(file1, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for line in csvreader:
        sent = line[0]
        label = line[1]
        sent = preprocess(sent.lower())
        sent.append("eos")
        if len(sent) == 0:
            continue
        samples.append(sent)
        tot_count += 1
        if label == 'question':
            labels.append(1)
            ques_count += 1
        else:
            labels.append(0)
word2idx = {}
idx2word = {}
word2idx['sos'] = 0
idx2word[0] = 'sos'
for sent in samples:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word
#import nltk and remove all functional words from the samples
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(len(stop_words))
stop_words = set(stopwords.words('english'))
and_words = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "if", "in", "is", "it", "of", "on", "or", "the",
    "to", "was", "with"
]
functional_words = [
    "about", "above", "across", "after", "against", "all", "almost", "along",
    "also", "although", "always", "am", "among", "an", "and", "another", "any",
    "anybody", "anyone", "anything", "anywhere", "apart", "are", "around", "as",
    "aside", "ask", "at", "away", "back", "be", "because", "before", "begin",
    "behind", "below", "between", "both", "but", "by", "can", "cannot", "certain",
    "choose", "come", "could", "do", "each", "either", "else", "ever", "every",
    "everybody", "everyone", "everything", "everywhere", "few", "find", "first",
    "for", "from", "get", "give", "go", "good", "have", "he", "her", "here", "hers",
    "him", "his", "how", "if", "in", "into", "is", "it", "its", "just", "last",
    "less", "like", "little", "look", "make", "many", "may", "me", "might", "more",
    "most", "much", "must", "my", "never", "no", "none", "nor", "not", "nothing",
    "now", "of", "off", "often", "on", "one", "only", "or", "other", "our", "ours",
    "out", "over", "own", "people", "put", "re", "said", "same", "see", "seem",
    "should", "so", "some", "someone", "something", "somewhere", "still", "such",
    "take", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "up", "use", "very",
    "want", "was", "way", "we", "well", "went", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your"
]
contractions = [
    "aint", "arent", "cant", "couldve", "couldnt", "didnt", "doesnt",
    "dont", "hadnt", "hasnt", "havent", "hed", "hell", "hes", "howd",
    "howll", "hows", "id", "ill", "im", "ive", "isnt", "itd", "itll",
    "its", "lets", "mightve", "mightnt", "mustve", "mustnt", "shant",
    "shed", "shell", "shes", "shouldve", "shouldnt", "thats", "theres",
    "theyd", "theyll", "theyre", "theyve", "wasnt", "wed", "well",
    "were", "weve", "werent", "whatll", "whatre", "whats", "wheres",
    "wholl", "whos", "whove", "wont", "wouldve", "wouldnt", "youd",
    "youll", "youre", "youve"
]
and_words = set(and_words)
functional_words = set(functional_words)
contractions = set(contractions)
#take union of all sets
stop_words = stop_words.union(and_words)
stop_words = stop_words.union(functional_words)
stop_words = stop_words.union(contractions)
print(len(stop_words))
# remove all functional words from the samples
ques_data = []
for i in range(len(samples)):
    if labels[i] == 1:
        samp = [w for w in samples[i] if not w in stop_words]
        ## add sos to beging of samp
        samp.insert(0, "sos")
        new_list = []
        new_list.append('sos')
        for k in samples[i]:
            new_list.append(k)
        loc_list = [samp, new_list]
        ques_data.append(loc_list)         
convert_dataset = Dataset(ques_data, word2idx, idx2word)
print(len(convert_dataset))
train_size = int(0.8 * len(convert_dataset))
test_size = len(convert_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(convert_dataset, [train_size, test_size])

n_epochs = 10
enc_model = EncoderRNN(300, 300).to(device)
dec_model = Attention_Decoder(300, len(word2idx)).to(device)
torch.save(word2idx, 'word2idx.pt')
torch.save(idx2word, 'idx2word.pt')
#SGD optimizer
# adam optimizer
optimizer = optim.Adam(list(enc_model.parameters()) + list(dec_model.parameters()), lr=0.0005)
criterion = nn.CrossEntropyLoss()
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
for epoch in range(n_epochs):
    avg_acc = 0
    cnt = 0
    for k, (x, y, idxs, strs) in enumerate(trainloader):
        # convert x and y to 2d
        a, b = x.shape[1], x.shape[2]
        x = x.view(a, b)
        a, b = y.shape[1], y.shape[2]
        y = y.view(a, b)
        # convert idxs also
        idxs = idxs.squeeze()
        x.to(device)
        y.to(device)
        hidden = torch.zeros(1, 300, device=device)
        x = x.float()
        hidden = hidden.float()
        # convert x and hidden to 3d
        x = x.unsqueeze(1)
        hidden = hidden.unsqueeze(0)
        encoder_output, encoder_hidden = enc_model(x, hidden)
        put_len = len(idxs)
        outputs = dec_model(encoder_output, encoder_hidden, put_len)
        # convert outputs items to 2d and then stack
        for i in range(len(outputs)):
            outputs[i] = outputs[i].squeeze()
        outputs = torch.stack(outputs)
        optimizer.zero_grad()
        loss = criterion(outputs, idxs)
        loss.backward()
        optimizer.step()
        out_class = []
        for i in range(len(outputs)):
            out_class.append(torch.argmax(outputs[i]).item())
        # given out_class and idxs, calculate accuracy
        acc = 0
        for i in range(len(out_class)):
            if out_class[i] == idxs[i]:
                acc += 1
        acc = acc / len(out_class)
        avg_acc += acc
        cnt += 1
        loc_acc = avg_acc / cnt
        print("loss: ", loss.item(), "acc: ", loc_acc, "epoch: ", epoch, "batch: ", k)
        
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
import gensim
print("loading embedddings...")
model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home2/advaith.malladi/GoogleNews-vectors-negative300.bin', binary=True)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, ques_data, word2idx, idx2word):
        self.ques_data = ques_data
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.in_vcts, self.out_vcts, self.out_idxs = self.get_vectors()
    def get_vectors(self):
        data = self.ques_data
        in_final = []
        out_final = []
        out_idx_final = []
        for row in data:
            input = row[0]
            output = row[1]
            in_vct = []
            out_vct = []
            out_idx = []
            for word in input:
                if word in model:
                    in_vct.append(model[word])
                else:
                    in_vct.append(np.zeros(300))
            for word in output:
                if word in model:
                    out_vct.append(model[word])
                else:
                    out_vct.append(np.zeros(300))
            for word in output:
                if word in self.word2idx:
                    out_idx.append(self.word2idx[word])
                else:
                    out_idx.append(self.word2idx['unk'])
            in_final.append(in_vct)
            out_final.append(out_vct)
            out_idx_final.append(out_idx)
        return in_final, out_final, out_idx_final
    def __len__(self):
        return len(self.ques_data)
    def __getitem__(self, index):
        return torch.tensor(self.in_vcts[index]).to(device), torch.tensor(self.out_vcts[index]).to(device), torch.tensor(self.out_idxs[index]).to(device), self.ques_data[index]
    
   
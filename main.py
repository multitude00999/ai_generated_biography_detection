# region Imports
import multiprocessing
import math
import time
import sys
import os
import re
import nltk
from util import extract_tokens, split_bios, clean_text
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import argparse
import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from nltk import word_tokenize
import gensim.downloader as api
from statistics import mean
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# endregion Imports

# region Constants
BIO_COL = "bio"
LABEL_COL = "label"
REAL_LABEL = 0
FAKE_LABEL = 1
SEED = 500
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
LEARNING_RATE = 5e-3
NUM_EPOCHS = 50
MODEL_PATH = 'lstm_classification_model.pt'
N_LAYERS = 2
BATCH_SIZE = 100
DROPOUT = 0.3
# endregion Constants

# region Setup & Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stop_words = list(set(stopwords.words('english')))
torch.manual_seed(SEED)
loaded_glove_model = api.load("glove-wiki-gigaword-300")
loaded_glove_embeddings = loaded_glove_model.vectors
# endregion Setup & Configurations

# region Data Loading
def read_data(lstPath):
    words = []
    for x in lstPath:
        tok = extract_tokens(x[0])
        words = words + split_bios(tok, x[1])
    df = pd.DataFrame(words, columns=[BIO_COL, LABEL_COL])
    df[BIO_COL] = df[BIO_COL].str.join(" ")
    df.drop_duplicates(inplace=True)
    return clean_text(df, BIO_COL, BIO_COL)
def split_bios(tokens, label):
    sb = eb = -1
    d = ""
    arr = []
    for i, x in enumerate(tokens):
        if x == "<start_bio>":
            sb = i
        elif x == "<end_bio>":
            eb = i
            if label == "":
                d = tokens[i + 1]
        if eb != -1 and sb != -1:
            if label == "":
                arr.append((tokens[sb + 1:eb], 0 if d == "[REAL]" else 1))
            else:
                arr.append((tokens[sb + 1:eb], label))
            eb = sb = -1
    return arr
def extract_tokens(fname):
    tokens = []
    with open(fname, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            for x in line.split(' '):
                if x != "":
                    tokens.append(x)
    return tokens
# endregion Data Loading

# region Data Pre-Processing
def clean_text(df, col, clean_col):
    st = SnowballStemmer('english')

    # change to lowercase and remove extra spaces on either ends
    df[clean_col] = df[col].apply(lambda x: x.lower().strip())

    # remove extra spaces between the words
    df[clean_col] = df[clean_col].apply(lambda x: re.sub(' +', ' ', x))

    # remove punctuation
    df[clean_col] = df[clean_col].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

    # remove stopwords and get the stem
    df[clean_col] = df[clean_col].apply(
        lambda x: ' '.join(st.stem(text) for text in x.split() if text not in stop_words))

    return df
# endregion Data Pre-Processing

# region Classification Dataset
class ClassificationDataset(Dataset):
    def __init__(self, data, labels, vocab=None, min_freq=5, max_length=300):
        self.data = data
        self.max_length = max_length
        if vocab is not None:  # imported
            self.word2idx, self.idx2word = vocab
        else:  # build if None is imported
            self.word2idx, self.idx2word = self.build_vocab(self.data, min_freq)
        # tokenization
        tokenized_data = [word_tokenize(bio.lower()) for bio in self.data]
        # list of indices
        indexed_data = [[self.word2idx.get(word, self.word2idx['UNK']) for word in bio] for bio in tokenized_data]
        # list of PyTorch LongTensors
        tensor_data = [torch.LongTensor(bio).to(device) for bio in indexed_data]
        # labels in a float tensor
        self.tensor_y = torch.FloatTensor(labels).to(device)
        # cut data to max_length
        cut_tensor_data = [tensor[:max_length] for tensor in tensor_data]
        # padding
        self.tensor_data = pad_sequence(cut_tensor_data, batch_first=True, padding_value=0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tensor_data[idx], self.tensor_y[idx]
    def build_vocab(self, corpus, threshold):
        vocab = {}
        for sent in corpus:
            for word in word_tokenize(sent.lower()):
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
        filtered_vocab = {word: count for word, count in vocab.items() if count >= threshold}
        words = sorted(filtered_vocab.keys(), key=vocab.get, reverse=True) + ['UNK']
        word_index = {words[i]: (i + 1) for i in range(len(words))}
        idx_word = {(i + 1): words[i] for i in range(len(words))}
        return word_index, idx_word
    def get_vocab(self):
        return self.word2idx, self.idx2word

# endregion Classification Dataset

# region Embeddings
def get_embeddings(glove_model, input_vocab):
  keys = {i: glove_model.vocab.get(w, None) for w, i in input_vocab.items()}
  index_dict = {i: key.index for i, key in keys.items() if key is not None}
  embeddings = np.zeros((len(input_vocab)+1,glove_model.vectors.shape[1]))
  for key, val in index_dict.items():
      embeddings[key] = glove_model.vectors[val]
  return embeddings
# endregion Embeddings

# Data Extraction for Bengio model
def decode(vocab,corpus):
    
    text = ''
    for i in range(len(corpus)):
        wID = corpus[i]
        text = text + vocab[wID] + ' '
    return(text)

def encode(words,text):
    corpus = []
    tokens = text.split(' ')
    for t in tokens:
        try:
            wID = words[t][0]
        except:
            wID = words['<unk>'][0]
        corpus.append(wID)
    return(corpus)

def read_encode(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
            
                    
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

def split_bio_bengio(test):
    indices = [i for i in range(len(test)) if test[i] == 121]
    x_rl = []
    y_rl = []
    x_fk = []
    y_fk = []
    rem = 0
    z=[]
    for i in indices:
        z.append([test[rem:i],test[i+3]])
        temp = z[-1][0]
        rem = i+5
        # print(test[i+3])
        if test[i+3] == vocab.index('[REAL]'):
            yy_x=[]
            yy_y=[]
            for i,bio in enumerate(temp):
                if i < len(temp) - CONTEXT_SIZE:
                    x_extract = []
                    for j in range(CONTEXT_SIZE):
                        x_extract.append(temp[i+j])
                    y_extract = [temp[i+5]]
                    yy_x.append(x_extract)
                    yy_y.append(y_extract)
            x_rl.append(yy_x)
            y_rl.append(yy_y)
        else:
            yy_x=[]
            yy_y=[]
            for i,bio in enumerate(temp):
                if i < len(temp) - CONTEXT_SIZE:
                    x_extract = []
                    for j in range(CONTEXT_SIZE):
                        x_extract.append(temp[i+j])
                    y_extract = [temp[i+CONTEXT_SIZE]]
                    yy_x.append(x_extract)
                    yy_y.append(y_extract)
            x_fk.append(yy_x)
            y_fk.append(yy_y)
    return x_rl,y_rl,x_fk,y_fk

def get_training_data_for_bengio(train,test,valid):
    x_train = []
    y_train = []
    for i,word_id in enumerate(train):
        if i < len(train) - CONTEXT_SIZE:
            x_extract = []
            for j in range(CONTEXT_SIZE):
                x_extract.append(train[i+j])
            y_extract = [train[i+CONTEXT_SIZE]]
            x_train.append(x_extract)
            y_train.append(y_extract)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = []
    y_test = []
    for i,word_id in enumerate(test):
        if i < len(test) - CONTEXT_SIZE:
            x_extract = []
            for j in range(CONTEXT_SIZE):
                x_extract.append(test[i+j])
            y_extract = [test[i+CONTEXT_SIZE]]
            x_test.append(x_extract)
            y_test.append(y_extract)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_valid = []
    y_valid = []
    for i,word_id in enumerate(valid):
        if i < len(valid) - CONTEXT_SIZE:
            x_extract = []
            for j in range(CONTEXT_SIZE):
                x_extract.append(valid[i+j])
            y_extract = [valid[i+CONTEXT_SIZE]]
            x_valid.append(x_extract)
            y_valid.append(y_extract)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    
    return x_train,y_train,x_valid,y_valid,x_test,y_test

# N-Gram Bengio Neural Network Model Region
class BengioNNmodel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(BengioNNmodel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h+(context_size * embedding_dim), vocab_size, bias = False)

    def forward(self, inputs):
        # compute x': concatenation of x1 and x2 embeddings
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
        # print(embeds.shape)
        # compute h: tanh(W_1.x' + b)
        out = torch.tanh(self.linear1(embeds))
        
        add_skip = torch.cat((out,embeds),1)
        # compute W_2.h
        out = self.linear2(add_skip)
        # compute y: log_softmax(W_2.h)
        log_probs = F.log_softmax(out, dim=1)
        # return log probabilities
        # BATCH_SIZE x len(vocab)
        return log_probs

# region Bengio Run
def run_FFNN(params):
    [vocab,words,train] = read_encode("mix.train.tok",[],{},[],3)
    [vocab,words,test] = read_encode("mix.test.tok",vocab,words,[],-1)
    [vocab,words,valid] = read_encode("mix.valid.tok",vocab,words,[],-1)
    gpu = 0 
    # word vectors size
    EMBEDDING_DIM = 200
    CONTEXT_SIZE = 6
    BATCH_SIZE = params.batch_size
    EPOCHS=params.epochs
    # hidden units
    H = 100
    torch.manual_seed(13013)
    
    x_train,y_train,x_valid,y_valid,x_test,y_test = get_training_data_for_bengio(train,test,valid)
    train_set = np.concatenate((x_train, y_train), axis=1)
    dev_set = np.concatenate((x_valid, y_valid), axis=1)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = available_workers)
    dev_loader = DataLoader(dev_set, batch_size = BATCH_SIZE, num_workers = available_workers)
    # create model
    loss_function = nn.CrossEntropyLoss()
    model = BengioNNmodel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, H)

    model=model.cuda(device)

    # using ADAM optimizer
    optimizer = optim.Adam(model.parameters(), lr = 2e-3)


    # ------------------------- TRAIN & SAVE MODEL ------------------------
    best_acc = 0
    best_model_path = None
    perp =[]
    for epoch in range(EPOCHS):
        st = time.time()
        print("\n--- Training model Epoch: {} ---".format(epoch+1))
        for it, data_tensor in enumerate(train_loader):       
            context_tensor = data_tensor[:,0:model.context_size]
            target_tensor = data_tensor[:,model.context_size]

            context_tensor, target_tensor = context_tensor.cuda(device), target_tensor.cuda(device)

            # zero out the gradients from the old instance
            model.zero_grad()

            # get log probabilities over next words
            log_probs = model(context_tensor)

            # calculate current accuracy
            acc = get_accuracy_from_log_probs(log_probs, target_tensor)

            # compute loss function
            loss = loss_function(log_probs, target_tensor)
            perplexity = np.exp2(loss.item())
            perp.append(perplexity)
            # backward pass and update gradient
            loss.backward()
            optimizer.step()

            if it % 500 == 0: 
                print("Training Iteration {} of epoch {} complete. Loss: {}; Acc:{}; Perplexity: {}; Time taken (s): {}".format(it, epoch, loss.item(), acc,perplexity, (time.time()-st)))
                st = time.time()

        # print("\n--- Evaluating model on dev data ---")
        dev_acc, dev_pre = evaluate_bengio(model, valid)
        # print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}".format(epoch, dev_acc, dev_pre))
        if dev_acc > best_acc:
            # print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            # set best model path
            best_model_path = 'bengio_{}-gram_model_{}.dat'.format(CONTEXT_SIZE,epoch)
            # saving best model
            torch.save(model.state_dict(), best_model_path)
    
    test_acc, test_pre = test_bengio(model, device)

# endregion FFNN Run

# Region Evaluate Bengio
# helper function to get accuracy from log probabilities
def get_accuracy_from_log_probs(log_probs, labels):
    probs = torch.exp(log_probs)
    predicted_label = torch.argmax(probs, dim=1)
    acc = (predicted_label == labels).float().mean()
    return acc

# helper function to evaluate model on dev data
def evaluate_bengio(model, valid):
    model.eval()
    x_vrl,y_vrl,x_vfk,y_vfk = split_bio_bengio(valid)

    with torch.no_grad():
        dev_st = time.time()
        available_workers = multiprocessing.cpu_count()
        lp =[]
        for i in range(len(x_vrl)):
            valid_rl_set = np.concatenate((x_vrl[i], y_vrl[i]), axis=1)
            valid_rl_loader = DataLoader(valid_rl_set, batch_size = 256, num_workers = available_workers)
            lp1 = []
            for it, data_tensor in enumerate(valid_rl_loader):
                context_tensor = data_tensor[:,0:model.context_size]
                target_tensor = data_tensor[:,model.context_size]
                context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
                log_probs = model(context_tensor)
                # lp1.append(float(log_probs[0][target_tensor]))
                # print(log_probs[0][0],target_tensor.shape)
                for i,x in enumerate(target_tensor):
                    lp1.append(float(log_probs[i][x]))
                del context_tensor
                del target_tensor
            lp.append(lp1)
        reallp=lp
        rl_avg = []
        for r in reallp:
            rl_avg.append(np.average(r))
        available_workers = multiprocessing.cpu_count()
        lp =[]
        for i in range(len(x_vrl)):
            valid_rl_set = np.concatenate((x_vrl[i], y_vrl[i]), axis=1)
            valid_rl_loader = DataLoader(valid_rl_set, batch_size = 256, num_workers = available_workers)
            lp1 = []
            for it, data_tensor in enumerate(valid_rl_loader):
                context_tensor = data_tensor[:,0:model.context_size]
                target_tensor = data_tensor[:,model.context_size]
                context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
                log_probs = model(context_tensor)
                # lp1.append(float(log_probs[0][target_tensor]))
                # print(log_probs[0][0],target_tensor.shape)
                for i,x in enumerate(target_tensor):
                    lp1.append(float(log_probs[i][x]))
                del context_tensor
                del target_tensor
            lp.append(lp1)
        fk_avg = []
        fakelp=lp
        for r in fakelp:
            fk_avg.append(np.average(r))    
    accuracy,precision,recall = metrics_bengio(rl_avg,fk_avg)
    acc = max(list(zip(*accuracy))[-1])
    pre = max(list(zip(*precision))[-1])
    return acc, pre

def test_bengio(model):
    model.eval()
    x_rl,y_rl,x_fk,y_fk = split_bio_bengio(test)
    for r in range(x_fk):
        if len(y_fk[r]) == 0:
            del x_fk[r]
            del y_fk[r]
    available_workers = multiprocessing.cpu_count()
    lp =[]
    for i in range(len(x_rl)):
        test_rl_set = np.concatenate((x_rl[i], y_rl[i]), axis=1)
        test_rl_loader = DataLoader(test_rl_set, batch_size = 256, num_workers = available_workers)
        lp1 = []
        for it, data_tensor in enumerate(test_rl_loader):
            context_tensor = data_tensor[:,0:xx.context_size]
            target_tensor = data_tensor[:,xx.context_size]
            context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
            log_probs = xx(context_tensor)
            # lp1.append(float(log_probs[0][target_tensor]))
            # print(log_probs[0][0],target_tensor.shape)
            for i,x in enumerate(target_tensor):
                lp1.append(float(log_probs[i][x]))
            del context_tensor
            del target_tensor
        lp.append(lp1)
    reallp=lp
    rl_avg = []
    for r in reallp:
        # rl_avg.append([np.average(r),len(r)])
        rl_avg.append(np.average(r))
    available_workers = multiprocessing.cpu_count()
    lp =[]
    for i in range(len(x_fk)):
        test_fk_set = np.concatenate((x_fk[i], y_fk[i]), axis=1)
        test_fk_loader = DataLoader(test_fk_set, batch_size = 512, num_workers = available_workers)
        lp1 = []
        for it, data_tensor in enumerate(test_fk_loader):
            context_tensor = data_tensor[:,0:xx.context_size]
            target_tensor = data_tensor[:,xx.context_size]
            context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
            log_probs = xx(context_tensor)
            # lp1.append(float(log_probs[0][target_tensor]))
            # print(log_probs[0][0],target_tensor.shape)
            for i,x in enumerate(target_tensor):
                lp1.append(float(log_probs[i][x]))
            del context_tensor
            del target_tensor
        lp.append(lp1)
    fk_avg = []
    fakelp=lp
    for r in fakelp:
        # fk_avg.append([np.average(r),len(r)])
        fk_avg.append(np.average(r))
    accuracy,precision,recall = metrics_bengio(rl_avg,fk_avg)
    plot_all_bengio(rl_avg,fk_avg,accuracy,precision,recall)

def plot_all_bengio(rl_avg,fk_avg,accuracy,precision,recall):
    rl_hist_counts, rl_hist_bins  = np.histogram(rl_avg)
    fk_hist_counts, fk_hist_bins = np.histogram(fk_avg)

    plt.figure()
    plt.hist(rl_hist_bins[:-1], rl_hist_bins, weights=rl_hist_counts,label=["Real","Fake"])
    plt.hist(fk_hist_bins[:-1], fk_hist_bins, weights=fk_hist_counts)

    plt.show()
    
    tt=list(zip(*accuracy))
    max1=max(tt[1])
    t=tt[0][tt[1].index(max1)]
    plt.figure()
    plt.plot(list(zip(*recall))[0], list(zip(*recall))[1] ,'.',color = 'red')
    plt.plot(list(zip(*precision))[0], list(zip(*precision))[1] ,'.',color = 'blue')
    plt.plot(list(zip(*accuracy))[0], list(zip(*accuracy))[1] ,'.',color = 'green')
    plt.text(-6, .2, "Precision", {'color': 'blue', 'fontsize': 13})
    plt.text(-6, .3, "Recall", {'color': 'red', 'fontsize': 13})
    plt.text(-6, .4, "Accuracy", {'color': 'green', 'fontsize': 13})

    plt.xlabel('Log Probability Threshold', fontsize=10)
    plt.axvline(x = t, color = 'b', label = 'treshold = {}'.format(t))
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'lower right')

    plt.show()
    
def metrics_bengio(rl_avg,fk_avg):
    tp=fp=tn=fn= 0
    y_min=min([min(rl_avg),min(fk_avg)])
    y_max=max([max(rl_avg),max(fk_avg)])
    yyy=0.1
    precision = []
    recall = []
    accuracy = []
    for ran in range(80):
        tp=fp=tn=fn= 0
        y_min+=yyy
        if y_min >= y_max:
            break
        for x in rl_avg:
            if x < y_min: 
                tp+=1
            else:
                fn+=1
        for x in fk_avg:
            if x >= y_min: 
                tn+=1
            else:
                fp+=1
        recall.append([y_min,tp/(tp+fn)])
        precision.append([y_min,tp/(tp+fp)])
        accuracy.append([y_min,(tp+tn)/(len(rl_avg)+len(fk_avg))])
    return accuracy,precision,recall
    

# region LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, vocabulary_size, hidden_dim, n_layers, gloveEmbeddings, embeddings=None,
                 fine_tuning=False, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        if embeddings:
            self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(gloveEmbeddings), freeze=not fine_tuning,
                                                           padding_idx=0)
        else:
            self.embeddings = nn.Embedding(num_embeddings=vocabulary_size + 1, embedding_dim=embedding_dim,
                                           padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, num_layers=n_layers,
                            dropout=dropout)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        emb = self.embeddings(inputs)
        lstm_out, (ht, ct) = self.lstm(emb, None)
        h = torch.cat((self.relu(ht[-2]), self.relu(ht[-1])), dim=1)
        x = torch.squeeze(self.linear(h))
        return x
# endregion LSTM Model

# region LSTM training
def train_epoch(model, opt, criterion, dataloader):
  model.train()
  losses = []
  perplexities = []
  accs = []
  for i, (x, y) in enumerate(dataloader):
      opt.zero_grad()
      # forward pass
      pred = model(x)
      # loss
      loss = criterion(pred, y)
      perplexity = 100 * np.exp2(loss.item())
      # backward pass
      loss.backward()
      # update weights
      opt.step()
      losses.append(loss.item())
      perplexities.append(perplexity)
      # accuracy
      num_correct = sum((torch.sigmoid(pred)>0.5) == y)
      acc = 100.0 * num_correct/len(y)
      accs.append(acc.item())
      if (i%20 == 0):
          print("Batch " + str(i) + " : training loss = " + str(loss.item()) + "; training acc = " + str(acc.item()))
  return losses, accs, perplexities
# endregion LSTM training

# region LSTM Evaluation
def eval_model(model, criterion, evalloader):
  model.eval()
  total_epoch_loss = 0
  total_epoch_perplexity = 0
  total_epoch_acc = 0
  preds = []
  with torch.no_grad():
      for i, (x, y) in enumerate(evalloader):
          pred = model(x)
          loss = criterion(pred, y)
          num_correct = sum((torch.sigmoid(pred)>0.5) == y)
          acc = 100.0 * num_correct/len(y)
          total_epoch_loss += loss.item()
          total_epoch_perplexity += (100 * np.exp2(loss.item()))
          total_epoch_acc += acc.item()
          preds.append(pred)

  return total_epoch_loss/(i+1), total_epoch_acc/(i+1), preds, total_epoch_perplexity/(i+1)
def evaluate(model, opt, criterion, training_dataloader, valid_dataloader, test_dataloader, num_epochs = 5):
  train_losses = []
  valid_losses = []
  train_perplexity = []
  valid_perplexity = []
  train_accs = []
  valid_accs = []

  print("Start Training...")
  for e in range(num_epochs):
      print("Epoch " + str(e+1) + ":")
      losses, acc, t_perplexity = train_epoch(model, opt, criterion, training_dataloader)
      train_losses.append(losses)
      train_perplexity.append(t_perplexity)
      train_accs.append(acc)
      valid_loss, valid_acc, val_preds, v_perplexity = eval_model(model, criterion, valid_dataloader)
      valid_losses.append(valid_loss)
      valid_perplexity.append(v_perplexity)
      valid_accs.append(valid_acc)
      print("Epoch " + str(e+1) + " : Validation loss = " + str(valid_loss) + "; Validation acc = " + str(valid_acc))
  test_loss, test_acc, test_preds, test_perplexity = eval_model(model, criterion, test_dataloader)
  print("Test loss = " + str(test_loss) + "; Test acc = " + str(test_acc))
  return train_losses, valid_losses, test_loss, train_accs, valid_accs, test_acc, test_preds, train_perplexity, valid_perplexity
# endregion LSTM Evaluation

# region LSTM Plots
def plot_LSTM_learning_curves(train_losses_lstm, valid_losses_lstm, train_accs_lstm,
                     valid_accs_lstm, train_perplexity, valid_perplexity):
    train_losses = [mean(train_loss) for train_loss in train_losses_lstm]
    train_accs = [mean(train_acc) for train_acc in train_accs_lstm]
    train_perplexity = [mean(tp) for tp in train_perplexity]
    epochs = [i for i in range(NUM_EPOCHS)]
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_accs, 'go-', label='Training Accuracy (LSTM)')
    ax[0].plot(epochs, valid_accs_lstm, 'ro-', label='validation Accuracy (LSTM)')
    ax[0].set_title('Training & Validation Accuracy (LSTM)')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_losses, 'go-', label='Training Loss (LSTM)')
    ax[1].plot(epochs, valid_losses_lstm, 'ro-', label='Validation Loss (LSTM)')
    ax[1].set_title('Training & Validation Loss (LSTM)')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    ax[2].plot(epochs, train_perplexity, 'go-', label='Training Perplexity (LSTM)')
    ax[2].plot(epochs, valid_perplexity, 'ro-', label='Validation Perplexity (LSTM)')
    ax[2].set_title('Perplexity (LSTM)')
    ax[2].legend()
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Perplexity")
    plt.show()

def plot_LSTM_confusion_matrix(test_preds_lstm, testY):
    preds = [(torch.sigmoid(t) > 0.5).tolist() for t in test_preds_lstm]
    preds = [int(t) for el in preds for t in el]
    cm = confusion_matrix(preds, testY)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['REAL', 'FAKE'],
                yticklabels=['REAL', 'FAKE'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(classification_report(testY, preds, target_names=['Predicted Fake', 'Predicted True']))

# endregion LSTM Plots

# region LSTM Load Saved Model
def load_LSTM(vocab_size):
    saved_model = LSTMClassifier(EMBEDDING_DIM, vocab_size, HIDDEN_DIM, embeddings=True, fine_tuning=False).to(device)
    saved_model.load_state_dict(torch.load(MODEL_PATH))
# endregion LSTM Load Saved Model

# region LSTM Run
def run_LSTM(params):
    train_files = [("mix.train.txt", ""), ("real.train.txt", REAL_LABEL), ("fake.train.txt", FAKE_LABEL)]
    df_train = read_data(train_files)
    df_train.drop_duplicates(inplace=True)
    trainX = df_train.iloc[:, :-1].squeeze()
    trainY = df_train.iloc[:, -1:].squeeze()

    valid_files = [("mix.valid.txt", ""), ("real.valid.txt", REAL_LABEL), ("fake.valid.txt", FAKE_LABEL)]
    df_valid = read_data(valid_files)
    df_valid.drop_duplicates(inplace=True)
    validX = df_valid.iloc[:, :-1].squeeze()
    validY = df_valid.iloc[:, -1:].squeeze()

    test_files = [("mix.test.txt", ""), ("real.test.txt", REAL_LABEL), ("fake.test.txt", FAKE_LABEL)]
    df_test = read_data(test_files)
    df_test.drop_duplicates(inplace=True)
    testX = df_test.iloc[:, :-1].squeeze()
    testY = df_test.iloc[:, -1:].squeeze()

    training_dataset = ClassificationDataset(trainX, trainY.values)
    training_word2idx, training_idx2word = training_dataset.get_vocab()
    valid_dataset = ClassificationDataset(validX, validY.values, (training_word2idx, training_idx2word))
    test_dataset = ClassificationDataset(testX, testY.values, (training_word2idx, training_idx2word))

    # arguments
    hidden_dim = params.hidden if params.hidden is not None and params.hidden > 0 else HIDDEN_DIM
    n_layers = params.n_layers if params.n_layers is not None and params.n_layers > 0 else N_LAYERS
    batch_size = params.batch_size if params.batch_size is not None and params.batch_size > 0 else BATCH_SIZE
    num_epochs = params.epochs if params.epochs is not None and params.epochs > 0 else NUM_EPOCHS
    lr = params.lr if params.lr is not None and params.lr > 0 else LEARNING_RATE
    dropout = params.dropout if params.dropout is not None and params.dropout > 0 else DROPOUT

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    gloveEmbeddings = get_embeddings(loaded_glove_model, training_word2idx)
    VOCAB_SIZE = len(training_word2idx)

    model_lstm = LSTMClassifier(EMBEDDING_DIM, VOCAB_SIZE, hidden_dim, n_layers,
                                gloveEmbeddings, embeddings=True, fine_tuning=False, dropout=dropout).to(device)
    opt = optim.Adam(model_lstm.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    train_losses_lstm, valid_losses_lstm, test_loss_lstm, train_accs_lstm, valid_accs_lstm, test_acc_lstm, \
    test_preds_lstm, train_perplexity, valid_perplexity = evaluate(model_lstm, opt, criterion, training_dataloader,
                                                                   valid_dataloader, test_dataloader, num_epochs)
    # save the model
    torch.save(model_lstm.state_dict(), MODEL_PATH)

    # plots
    plot_LSTM_learning_curves(train_losses_lstm, valid_losses_lstm, train_accs_lstm,
                     valid_accs_lstm, train_perplexity, valid_perplexity)

    plot_LSTM_confusion_matrix(test_preds_lstm, testY)

# endregion LSTM Run

# region Main
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-d_hidden', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-seq_len', type=int, default=30)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-dropout', type=int, default=0.35)
    parser.add_argument('-clip', type=int, default=2.0)
    parser.add_argument('-model', type=str, default='LSTM')
    parser.add_argument('-savename', type=str, default='lstm')
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-trainname', type=str)
    parser.add_argument('-validname', type=str)
    parser.add_argument('-testname', type=str)

    params = parser.parse_args()
    torch.manual_seed(SEED)

    if params.model == 'FFNN':
        run_FFNN(params))
    elif params.model == 'LSTM':
        run_LSTM(params)


if __name__ == "__main__":
    main()

# endregion Main
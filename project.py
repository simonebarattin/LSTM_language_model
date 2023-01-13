import torch
import random
import math
import numpy as np
from torch import nn
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Implementation of regularization techniques (WeightDrop, LockedDrop, EmbeddDropout) from official paper implementation https://github.com/salesforce/awd-lstm-lm

TRAIN_DATA_PATH = "ptbdataset/ptb.train.txt"
VAL_DATA_PATH = "ptbdataset/ptb.valid.txt"
TEST_DATA_PATH = "ptbdataset/ptb.test.txt"

save_path = "best_model.pth"

def load_data_tokenize(path):
    tokens = []
    with open(path, 'r') as f:
        corpus = f.readlines()
    for line in corpus:
        tokens.extend(line.split() + ['<eos>'])
    return tokens

def save_model(model, path):
    torch.save({
        "state_dict": model.state_dict()
    }, path)

def load_model(path):
    checkpoint = torch.load(path)
    return checkpoint

class Vocabulary():
    def __init__(self):
        self.num_tokens = 0
        self.word2idx = {}
        self.idx2word = {}

    def add2vocab(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_tokens
            self.idx2word[self.num_tokens] = word
            self.num_tokens += 1

    def process_tokens(self, tokens):        
        for token in tokens:
            self.add2vocab(token)

    def __len__(self):
        return len(self.idx2word)

# data usage from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/data_utils.py
class Dataset():
    def __init__(self, vocab, tokens, batch_size) -> None:
        super(Dataset, self).__init__()
        self.idx2word = vocab.idx2word
        self.word2idx = vocab.word2idx
        self.batch_size = batch_size
        self.tokens = tokens
        self.data = torch.LongTensor(len(tokens))
        for count, t in enumerate(self.tokens):
            self.data[count] = self.word2idx[t]
        num_batches = self.data.size(0) // batch_size
        self.data = self.data[:num_batches*batch_size]
        self.data = self.data.view(batch_size, -1).t().contiguous()
        if torch.cuda.is_available():
            self.data.cuda()

    def get_batch(self, idx, seq_len):
        # x = self.data[:, idx:idx+seq_len]
        x = self.data[idx:idx+seq_len]
        # y = self.data[:, idx+1:idx+seq_len+1]
        y = self.data[idx+1:idx+seq_len+1]
        return x, y

# DropConnect on recurrent hidden to hidden weight matrices https://github.com/salesforce/awd-lstm-lm
class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        for w_name in self.weights:
            w = getattr(self.module, w_name)
            del self.module._parameters[w_name]
            self.module.register_parameter(w_name+"_raw", nn.Parameter(w.data))

    def _setweight(self):
        for w_name in self.weights:
            w_raw = getattr(self.module, w_name+"_raw")
            new_w = nn.functional.dropout(w_raw, self.dropout)
            setattr(self.module, w_name, new_w)

    def forward(self, *args):
        self._setweight()
        return self.module.forward(*args)

# uses a unique dropout mask for different samples, which stays the same within the forward and backward pass https://github.com/salesforce/awd-lstm-lm
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout, training):
        # create a binary mask of 0 and 1 using bernoulli
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1-dropout)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1-dropout)
        mask = mask.expand_as(x)
        if training:
            return mask * x
        else:
            return x

# https://github.com/salesforce/awd-lstm-lm
class EmbeddDropout(nn.Module):
    def __init__(self, dropout):
        super(EmbeddDropout, self).__init__()
        self.dropout = dropout

    def forward(self, embedding, words, training):
        mask = embedding.weight.data.new().resize_((embedding.weight.size(0), 1)).bernoulli_(1-self.dropout).expand_as(embedding.weight) / (1-self.dropout)
        masked_emb = embedding.weight * mask if self.dropout and training else embedding.weight
        padding_idx = embedding.padding_idx 
        if padding_idx is None: 
            padding_idx = -1
        x = nn.functional.embedding(words, masked_emb, padding_idx, embedding.max_norm, embedding.norm_type, embedding.scale_grad_by_freq, embedding.sparse)
        return x

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, dropoute, dropoutw, dropouti, dropouth, tweights=True, attention=False):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size 
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.tweights = tweights
        self.attention = attention

        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoutw = dropoutw
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.edrop = EmbeddDropout(dropoute)
        self.lockeddrop = LockedDropout()
        # Create LSTM with for instead of using n_layers so that we can apply Weight Drop between each hh matrix
        lstms = []
        for i in range(n_layers):
            inp = embedding_dim if i==0 else hidden_dim
            hid = hidden_dim if i!=n_layers-1 else (embedding_dim if tweights else hidden_dim)
            lstms.append(nn.LSTM(inp, hid, 1, dropout=0))
        self.lstm = [WeightDrop(lstm, ["weight_hh_l0"], dropoutw) for lstm in lstms]
        self.lstm = nn.ModuleList(self.lstm)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention_fc = nn.Linear(((self.n_layers-1)*self.hidden_dim+self.embedding_dim)*2, vocab_size)
        self.att1 = nn.Linear()

        if self.tweights:
            self.fc.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range, init_range)

        init_range = 1 / math.sqrt(self.hidden_dim)
        for i, lstm in enumerate(self.lstm):
            lstm.module._all_weights[0][0] = torch.FloatTensor(
                self.embedding_dim if i==0 else self.hidden_dim,
                self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).uniform_(-init_range, init_range)
            lstm.module._all_weights[0][1] = torch.FloatTensor(
                self.embedding_dim if i==0 else self.hidden_dim,
                self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).uniform_(-init_range, init_range)

    def forward(self, x, hidden):
        b_s = x.shape[0]
        x = self.edrop(self.embedding, x, self.training)
        x = self.lockeddrop(x, self.dropouti, self.training)

        new_h, hid, cont = [], [], []
        for i in range(self.n_layers):
            out, h = self.lstm[i](x, hidden[i])
            hid.append(h[0].squeeze(0))
            new_h.append(h)
            x = out
            if i != n_layers-1:
                out = self.lockeddrop(out, self.dropouth, self.training)
            if self.attention:
                weights = torch.bmm(out.permute(1,0,2), h[0].permute(1,2,0)).squeeze(-1)
                print(weights.shape)
                sw = torch.nn.functional.softmax(weights, 1)
                print(sw.shape)
                context = torch.bmm(out.permute(1,2,0), sw.unsqueeze(-1)).squeeze(-1)
                print(context.shape)
                cont.append(context)
        if self.attention:
            hid = torch.cat(hid, 1)
            print(hid.shape)
            cont = torch.cat(cont, 1)
            print(cont.shape)
            combined = torch.cat((hid, cont), 1)
            print(combined.shape)
            quit()
            out = self.attention_fc(combined)
            return out, new_h
        out = self.lockeddrop(out, self.dropout, self.training)
        print(out.shape)
        quit()
        out = out.view(out.size(0)*out.size(1), out.size(2))
        out = self.fc(out)

        return out, new_h

    def init_hidden(self, batch_size):
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        hidden = [(weights.new(1, batch_size, self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).zero_().cuda(), 
                    weights.new(1, batch_size, self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).zero_().cuda()) 
                    for i in range(self.n_layers)]
        
        return hidden

# from professor notebook
# def init_weights(mat):
#         for m in mat.modules():
#             if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
#                 for name, param in m.named_parameters():
#                     if 'weight_ih' in name:
#                         for idx in range(4):
#                             mul = param.shape[0]//4
#                             torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
#                     elif 'weight_hh' in name:
#                         for idx in range(4):
#                             mul = param.shape[0]//4
#                             torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
#                     elif 'bias' in name:
#                         param.data.fill_(0)
#             else:
#                 if type(m) in [nn.Linear]:
#                     torch.nn.init.uniform_(m.weight, -0.01, 0.01)
#                     if m.bias != None:
#                         m.bias.data.fill_(0.01)
    
# def batchify(vocab, tokens, seq_len, batch_size):
#     num_tokens = []
#     for t in tokens:
#         num_tokens.append(vocab.word2idx[t])
#     n_batches = len(num_tokens) // batch_size
#     x, y = [], []
#     num_tokens = num_tokens[:n_batches*batch_size]

#     for idx in range(0, len(num_tokens)-seq_len):
#         end = idx + seq_len
#         batched_data = num_tokens[idx:end]
#         x.append(batched_data)
#         y.append(num_tokens[idx+1:end+1])
    
#     return torch.tensor(x), torch.tensor(y)

def detach_hidden(hidden):
    if len(hidden) > 0:
        detached = []
        for h in hidden:
            detached.append(tuple([h[0].detach(), h[1].detach()]))
    else:
        detached = tuple([h[0].detach(), h[1].detach()])
    return detached

epochs = 100
batch_size = 20 #80 #512
test_batch_size = 10
seq_len = 70 #10
seq_len_threshold = 0.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 30
momentum = 0.9
weight_decay = 1e-6

embedding_dimension = 400 #512
hidden_dimension = 1150 #512
n_layers = 3 #1

# Regularization
clip_grad = 0.25
dropout = 0.5
dropout_emb = 0.3
dropout_inp = 0.3
dropout_hid = 0.5
dropout_wgt = 0.5
tye_weights = True

# early stopping
patience = 5
best_loss = float('inf')

train_tokens = load_data_tokenize(TRAIN_DATA_PATH)
val_tokens = load_data_tokenize(VAL_DATA_PATH)
test_tokens = load_data_tokenize(TEST_DATA_PATH)

print("Checking OOV words")
train_set = set(train_tokens)
val_set = set(val_tokens)
test_set = set(test_tokens)
print("Test-Train: ",len(test_set.difference(train_set)), " Val-Train: ", len(val_set.difference(train_set)))

vocab = Vocabulary()
vocab.add2vocab("<unk>")
vocab.add2vocab("<eos>")
vocab.process_tokens(train_tokens)

##################### TEST NOT DATALOADER #####################
train_ids = Dataset(vocab, train_tokens, batch_size)
val_ids = Dataset(vocab, val_tokens, batch_size)
test_ids = Dataset(vocab, test_tokens, test_batch_size)
##############################################################

model = LSTMModel(len(vocab), embedding_dimension, hidden_dimension, n_layers, dropout, dropout_emb, dropout_wgt, dropout_inp, dropout_hid, attention=True).to(device)
# model.apply(init_weights)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) # following paper we use SGD without momentum
asgd = False
criterion = nn.CrossEntropyLoss()
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True)

writer = SummaryWriter()

best_ppl = float('inf')
for epoch in range(epochs):
    hidden = model.init_hidden(batch_size)
    losses = []
    ppls = []
    model.train()
    # apply shuffling to the batches of sequences
    # rnd_idxs = list(range(0, train_ids.data.size(1) - seq_len, seq_len))
    # random.shuffle(rnd_idxs)
    mu = seq_len if random.random() < seq_len_threshold else seq_len/2
    std = 5
    # for i in tqdm(rnd_idxs):
    # for i in tqdm(range(0, train_ids.data.size(1) - seq_len, seq_len)):
    batch, i = 0, 0
    # use while since bptt changes at each step
    while i < train_ids.data.size(0) -1 -1:
        # following the paper, sample different sequence lengths in order to learn throughout all the corpus
        bptt = max(std, int(np.random.normal(mu, std)))
        new_lr = (lr * bptt) / mu
        optimizer.param_groups[0]['lr'] = new_lr

        x, y = train_ids.get_batch(i, bptt)
        if x.shape!=y.shape:
            break
        x = x.to(device)
        y = y.to(device)
        h = detach_hidden(hidden)

        optimizer.zero_grad()
        output, h = model(x, h)

        # output = output.reshape(batch_size * seq_len, -1)
        y = y.reshape(-1)

        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        losses.append(loss.item())
        cur_ppl = np.exp(loss.item())
        ppls.append(cur_ppl)
        i += bptt

    cur_loss = np.mean(losses)
    cur_ppl = np.exp(cur_loss)
    writer.add_scalar("Loss/train", cur_loss, epoch)
    writer.add_scalar("Perplexity/train", cur_ppl, epoch)
    writer.flush()
    print(f"Train epoch {epoch}")
    print(f"\tLoss: {cur_loss}, PPL: {cur_ppl}")

    hidden = model.init_hidden(batch_size)
    losses = []
    ppls = []
    model.eval()
    with torch.no_grad():
        # for i in tqdm(range(0, val_ids.data.size(1) - seq_len, seq_len)):
        for i in tqdm(range(0, val_ids.data.size(0) - 1, seq_len)):
            x, y = val_ids.get_batch(i, seq_len)
            if x.shape!=y.shape:
                continue
            x = x.to(device)
            y = y.to(device)
            h = detach_hidden(hidden)
            # h = tuple([each.data for each in hidden])

            output, h = model(x, h)
            # output = output.reshape(batch_size * seq_len, -1)
            y = y.reshape(-1)

            loss = criterion(output, y)
            cur_ppl = np.exp(loss.item())

            losses.append(loss.item())
            ppls.append(np.exp(loss.item()))

        cur_loss = np.mean(losses)
        cur_ppl = np.exp(cur_loss)
        writer.add_scalar("Loss/validation", cur_loss, epoch)
        writer.add_scalar("Perplexity/validation", cur_ppl, epoch)
        writer.flush()
        # lr_scheduler.step(cur_loss)
        print(f"Validation epoch {epoch}")
        print(f"\tLoss: {cur_loss}, PPL: {cur_ppl}")
        if cur_ppl < best_ppl:
            print(f"Save model at epoch {epoch} with best perplexity {cur_ppl}")
            save_model(model, save_path)
            best_ppl = cur_ppl
        if cur_loss >= best_loss:
            patience -= 1
            best_loss = cur_loss
            print(f"Loss not decreasing... Patience to {patience}")
            if not asgd:
                asgd = True
                optimizer = torch.optim.ASGD(model.parameters(), lr=lr, lambd=0., weight_decay=weight_decay, t0=0)
                print("\tSwitching to ASGD with lr ", lr)
            if patience == 0:
                print("Training stopped due to early stopping!")
                break
        else:
            best_loss = cur_loss
            patience = 5

hidden = model.init_hidden(test_batch_size)
losses = []
ppls = []
model.eval()
with torch.no_grad():
    for i in tqdm(range(0, test_ids.data.size(0) - seq_len, seq_len)):
        x, y = test_ids.get_batch(i, seq_len)
        x = x.to(device)
        y = y.to(device)
        # h = tuple([each.data for each in hidden])
        h = detach_hidden(hidden)

        output, h = model(x, h)
        # output = output.reshape(test_batch_size * seq_len, -1)
        y = y.reshape(-1)

        loss = criterion(output, y)
        cur_ppl = np.exp(loss.item())

        losses.append(loss.item())
        ppls.append(np.exp(loss.item()))

    cur_loss = np.mean(losses)
    cur_ppl = np.exp(cur_loss)
    writer.add_scalar("Loss/test", cur_loss, epoch)
    writer.add_scalar("Perplexity/test", cur_ppl, epoch)
    writer.flush()
    print(f"Test")
    print(f"\tLoss: {cur_loss}, PPL: {cur_ppl}")
writer.close()
    
def generate(prompt, max_seq_len, temperature, model, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = prompt.split()
    indices = [vocab.word2idx[t] for t in tokens]
    inp = torch.LongTensor([indices]).to(device)
    batch_size = 1
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(max_seq_len):
            prediction, hidden = model(inp, hidden)
            probs = prediction.squeeze().data.div(temperature).exp()
            prediction = torch.multinomial(probs, num_samples=1).item()
            
            while prediction == vocab.word2idx['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab.word2idx['<eos>']:
                break
            
            indices.append(prediction)
            inp.data.fill_(prediction)

    tokens = [vocab.idx2word[i] for i in indices]
    return tokens

ckpt = load_model(save_path)
model.load_state_dict(ckpt['state_dict'])
prompt = 'the'
max_seq_len = 30
seed = 0

temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
for temperature in temperatures:
    generation = generate(prompt, max_seq_len, temperature, model, vocab, device, seed)
    print(str(temperature)+'\n'+' '.join(generation)+'\n')
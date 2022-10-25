import torch
import numpy as np
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

TRAIN_DATA_PATH = "ptbdataset/ptb.train.txt"
VAL_DATA_PATH = "ptbdataset/ptb.valid.txt"
TEST_DATA_PATH = "ptbdataset/ptb.test.txt"

def tokenize(text, vocab):
    res = []
    for t in text:
        tokenized = t.split()
        res.extend(tokenized + ["<eos>"])
        for tok in tokenized:
            vocab.add2vocab(tok)
    return res

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

    def __len__(self):
        return len(self.idx2word)

class Dataset(data.Dataset):
    def __init__(self, vocab, tokens, batch_size, seq_len) -> None:
        super(Dataset, self).__init__()
        self.idx2word = vocab.idx2word
        self.word2idx = vocab.word2idx
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokens = tokens
        self.num_tokens = []
        for t in self.tokens:
            self.num_tokens.append(self.word2idx[t])

        self.data, self.targets = self.batchify(self.num_tokens, seq_len, batch_size)

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def batchify(self, tokens, seq_len, batch_size):
        n_batches = len(tokens) // batch_size
        x, y = [], []
        tokens = tokens[:n_batches*batch_size]

        for idx in range(0, len(tokens)-seq_len):
            end = idx + seq_len
            batched_data = tokens[idx:end]
            x.append(batched_data)
            y.append(tokens[end])
        
        return torch.tensor(x), torch.tensor(y)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size 
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        b_s = x.shape[0]
        x = self.embedding(x)    

        out, (final_hidden, cells) = self.lstm(x)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = out.view(b_s, -1, self.output_size)

        return out, (final_hidden, cells)

    def init_hidden(self, batch_size):
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                    weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        
        return hidden

def init_weights(mat):
        for m in mat.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                    elif 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            else:
                if type(m) in [nn.Linear]:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)
    
batch_size = 128
seq_len = 10
device = "cuda:0"

lr = 0.001
epochs = 10
embedding_dimension = 200
hidden_dimension = 250
clip_grad = 5
n_layers = 1
dropout = 0

vocab = Vocabulary()
vocab.add2vocab("<eos>")
with open(TRAIN_DATA_PATH, 'r') as f:
    lines = f.readlines()
tokenized = tokenize(lines, vocab)
output_size = len(vocab)

dataset = Dataset(vocab, tokenized, batch_size, seq_len)
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = LSTMModel(len(vocab), output_size, embedding_dimension, hidden_dimension, n_layers, dropout).to(device)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    hidden = model.init_hidden(batch_size)
    losses = []
    model.train()
    for batch_idx, (x,y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        h = tuple([each.data for each in hidden])

        optimizer.zero_grad()
        output, h = model(x, h)

        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        losses.append(loss.item())
    print(f"Train epoch {epoch}")
    print(f"\tLoss: {np.mean(losses)}, PPL: {np.exp(np.mean(losses))}")

    # losses = []
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, (x,y) in enumerate(tqdm(loader)):
    #         x = x.to(device)
    #         y = y.to(device)
    #         h = tuple([each.data for each in hidden])

    #         optimizer.zero_grad()
    #         output, h = model(x, h)

    #         loss = criterion(output, y)
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    #         optimizer.step()
    #         losses.append(loss.item())
    #     print(f"Validation epoch {epoch}")
    #     print(f"\tLoss: {np.mean(losses)}, PPL: {np.exp(np.mean(losses))}")
    
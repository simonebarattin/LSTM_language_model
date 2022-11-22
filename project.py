import torch
import numpy as np
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

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
    torch.save(model.state_dict(), path)

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
        self.data = self.data.view(batch_size, -1)

    def get_batch(self, idx, seq_len):
        x = self.data[:, idx:idx+seq_len]
        y = self.data[:, idx+1:idx+seq_len+1]
        return x, y
        # self.data, self.targets = self.batchify(self.num_tokens, seq_len, batch_size)
        

    # def __len__(self):
    #     return len(self.tokens)
    
    # def __getitem__(self, idx):
    #     return self.data[idx], self.targets[idx]

    # def batchify(self, tokens, seq_len, batch_size):
    #     n_batches = len(tokens) // batch_size
    #     x, y = [], []
    #     tokens = tokens[:n_batches*batch_size]

    #     for idx in range(0, len(tokens)-seq_len):
    #         end = idx + seq_len
    #         batched_data = tokens[idx:end]
    #         x.append(batched_data)
    #         y.append(tokens[end])
        
    #     return torch.tensor(x), torch.tensor(y)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size 
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        b_s = x.shape[0]
        x = self.embedding(x)    

        out, (final_hidden, cells) = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = out.view(b_s, -1, self.vocab_size)

        return out, (final_hidden, cells)

    def init_hidden(self, batch_size):
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                    weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        
        return hidden

# from professor notebook
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
    
def batchify(vocab, tokens, seq_len, batch_size):
    num_tokens = []
    for t in tokens:
        num_tokens.append(vocab.word2idx[t])
    n_batches = len(num_tokens) // batch_size
    x, y = [], []
    num_tokens = num_tokens[:n_batches*batch_size]

    for idx in range(0, len(num_tokens)-seq_len):
        end = idx + seq_len
        batched_data = num_tokens[idx:end]
        x.append(batched_data)
        y.append(num_tokens[idx+1:end+1])
    
    return torch.tensor(x), torch.tensor(y)

batch_size = 128
test_batch_size = 10
seq_len = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
epochs = 10
embedding_dimension = 512
hidden_dimension = 512
clip_grad = 0.25
n_layers = 1
dropout = 0

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

# train_x, train_y = batchify(vocab, train_tokens, seq_len, batch_size)
# train_dataset = data.TensorDataset(train_x, train_y)
# train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_x, val_y = batchify(vocab, val_tokens, seq_len, batch_size)
# val_dataset = data.TensorDataset(val_x, val_y)
# val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# test_x, test_y = batchify(vocab, test_tokens, seq_len, batch_size)
# test_dataset = data.TensorDataset(test_x, test_y)
# test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

##################### TEST NOT DATALOADER #####################
train_ids = Dataset(vocab, train_tokens, batch_size)
val_ids = Dataset(vocab, val_tokens, batch_size)
test_ids = Dataset(vocab, test_tokens, test_batch_size)
##############################################################

model = LSTMModel(len(vocab), embedding_dimension, hidden_dimension, n_layers, dropout).to(device)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

best_ppl = float('inf')
for epoch in range(epochs):
    hidden = model.init_hidden(batch_size)
    losses = []
    ppl = []
    model.train()
    # for batch_idx, (x,y) in enumerate(tqdm(train_loader)):
    for i in range(0, train_ids.size(1) - seq_len, seq_len):
        x, y = train_ids.get_batch(i, seq_len)
        x = x.to(device)
        y = y.to(device)
        h = tuple([each.data for each in hidden])

        optimizer.zero_grad()
        output, h = model(x, h)

        output = output.reshape(batch_size * seq_len, -1)
        y = y.reshape(-1)

        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        losses.append(loss.item())
        ppl.append(np.exp(loss.item()))
    print(f"Train epoch {epoch}")
    print(f"\tLoss: {np.mean(losses)}, PPL: {np.exp(np.mean(losses))}, PPL per word: {np.mean(ppl)}")

    losses = []
    ppl = []
    model.eval()
    with torch.no_grad():
        # for batch_idx, (x,y) in enumerate(tqdm(val_loader)):
        for i in range(0, val_ids.size(1) - seq_len, seq_len):
            x, y = val_ids.get_batch(i, seq_len)
            x = x.to(device)
            y = y.to(device)
            h = tuple([each.data for each in hidden])

            output, h = model(x, h)
            output = output.reshape(batch_size * seq_len, -1)
            y = y.reshape(-1)

            loss = criterion(output, y)
            losses.append(loss.item())
            ppl.append(np.exp(loss.item()))
        print(f"Validation epoch {epoch}")
        print(f"\tLoss: {np.mean(losses)}, PPL: {np.exp(np.mean(losses))}, PPL per word: {np.mean(ppl)}")
        if np.exp(np.mean(losses)) < best_ppl:
            print(f"Save model at epoch {epoch} with best perplexity {np.exp(np.mean(losses))}")
            save_model(model, save_path)
    
import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(VanillaLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.model = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h):
        b_s = x.shape[0]
        x = self.embedding(x)    

        x, h = self.model(x)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.fc(x)
        # x = x.view(b_s, -1, self.vocab_size)
        return x, h

    def detach_h(self, h):
        hidden, cell = h
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def init_hidden(self, batch_size, use_cuda):
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        if use_cuda:
            hidden = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                        weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_(), 
                        weights.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden
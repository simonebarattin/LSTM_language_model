import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    '''
        Attention module which takes the LSTM output and creates an attention map. This is multiplied by the output to obtain the result 
        of the soft attention. The code was inspired by [1]

        References:
            [1] https://github.com/ap229997/LanguageModel-using-Attention/blob/a6a3179b8225ec1649e0e17f111db721d64357e2/model/attention.py
    '''
    def __init__(self, seq_len, hidden_dim, dropout=0.5):
        super(AttentionLayer, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.mlp1_dim = hidden_dim*3
        self.mlp2_dim = hidden_dim
        self.dropout = dropout

        self.fcs = nn.Sequential(
            nn.Linear(seq_len*hidden_dim, self.mlp1_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.mlp1_dim, self.mlp2_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.mlp2_dim, seq_len),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_emb):
        bs = lstm_emb.shape[1]
        lstm_emb = lstm_emb.contiguous()
        lstm_emb_flat = lstm_emb.view(bs, -1)

        attention = self.fcs(lstm_emb_flat)
        alpha = self.softmax(attention)
        alpha = torch.stack([alpha]*self.mlp2_dim, dim=2)
        alpha = alpha.transpose(0,1)

        attention_map = lstm_emb * alpha
        # attention_map = torch.sum(attention_map, dim=0, keepdim=True)
        return attention_map

class Attention_LSTM(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim, hidden_dim, num_layers, dropout=0.5, tyeweights=False):
        super(Attention_LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = dropout
        self.tweights = tyeweights

        self.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.attention = AttentionLayer(seq_len, hidden_dim)
        self.FC = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        emb = self.Embedding(x)
        lstm_emb, h = self.LSTM(emb, hidden)

        attention_emb = self.attention(lstm_emb)
        bs = attention_emb.shape[1]
        attention_emb = attention_emb.contiguous().view(-1, self.hidden_dim)
        attention_emb = self.dropout(attention_emb)

        out = self.FC(attention_emb)
        return out, h

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
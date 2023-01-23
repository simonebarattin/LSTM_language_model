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
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp1_dim, seq_len),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_emb):
        bs = lstm_emb.shape[1]
        lstm_emb = lstm_emb.contiguous()
        lstm_emb_flat = lstm_emb.view(bs, -1)

        attention = self.fcs(lstm_emb_flat)
        alpha = self.softmax(attention)
        alpha = torch.stack([alpha]*self.hidden_dim, dim=2)
        alpha = alpha.transpose(0,1)

        attention_map = lstm_emb * alpha
        attention_map = torch.sum(attention_map, dim=0, keepdim=True)
        attention_map = attention_map.squeeze()
        attention_map = torch.stack([attention_map]*self.seq_len, dim=0)
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

import torch
import torch.nn as nn

class LSTMWithAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers=1, attention_type='multiplicative'):
        super(LSTMWithAttention, self).__init__()
        self.embedding_dim = embedding_size
        self.hidden_dim = hidden_size
        self.vocab_dim = vocab_size
        self.num_layers = num_layers
        self.attention_type = attention_type

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.attention_type = attention_type
        if attention_type == 'dot':
            self.attention = DotProductAttention(hidden_size)
        elif attention_type == 'multiplicative':
            self.attention = MultiplicativeAttention(hidden_size)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, vocab_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        input_emb = self.embedding(input)
        lstm_output, hidden = self.lstm(input_emb, hidden)
        attention_output = self.attention(lstm_output, lstm_output, lstm_output)
        # attention_weights = self.attention(lstm_output)
        # weighted_output = lstm_output * attention_weights
        output = self.output_layer(attention_output)

        output = output.view(output.size(0)*output.size(1), output.size(2))
        return output, hidden

    def init_hidden(self, batch_size, use_cuda):
        weights = next(self.parameters()).data
        if use_cuda:
            hidden = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                        weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_(), 
                        weights.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden

class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(DotProductAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, lstm_output):
        attention_weights = lstm_output.dot(lstm_output.transpose(1, 2))
        return attention_weights

class MultiplicativeAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiplicativeAttention, self).__init__()
        self.hidden_size = hidden_size
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1)

    def forward(self, lstm_output):
        key = self.key_layer(lstm_output)
        query = self.query_layer(lstm_output)
        value = self.value_layer(lstm_output)
        # attention_weights = key.dot(query.transpose(1, 2))
        (output, attention_weigths)= self.attention(query, key, value)

        return attention_weigths.transpose(1,0)

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = hidden_size
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.query_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_output):
        key = self.key_layer(lstm_output)
        query = self.query_layer(lstm_output)
        attention_weights = key + query
        return attention_weights

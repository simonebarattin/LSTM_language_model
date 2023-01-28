import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class AT_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers=1, bidirectional=False):
        super(AT_LSTM, self).__init__()
        self.embedding_dim = embedding_size
        self.vocab_dim = vocab_size
        self.num_layers = num_layers
        self.bidirectional = 2 if bidirectional else 1 
        self.hidden_dim = hidden_size * self.bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        
        self.attention = nn.MultiheadAttention(hidden_size, 1)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        input_emb = self.embedding(input)
        lstm_output, hidden = self.lstm(input_emb, hidden)
        attention_output, attention_weights = self.attention(lstm_output, lstm_output, lstm_output)

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
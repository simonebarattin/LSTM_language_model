import torch
import torch.nn as nn

'''
    Script to define an attention mechanism. The attention mechanism is the Bahdanau attention explained in [1]

    Args:
        hidden_dim (int)    : input dimension to the linear layer, i.e. the hidden dimension of the LSTM
        dropout (float)     : value of the dropout
    
    Return:
        attention_weights (torch.FloatTensor) : tensor of weights from the attention mechanism

    Reference:
        [1] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” 2014
'''
class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(AttentionMechanism, self).__init__()
        self.attention = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weigths = self.attention(x)
        return attention_weigths

'''
    Script to define an LSTM with an attention mechanism.

    Args:
        embedding_size (int)    : size of the embedding
        hidden_size (int)       : size of the hidden dimension
        vocab_size (int)        : size of the vocabulary, i.e. number of words in the vocabulary
        num_layers (int)        : number of LSTM layers
        bidirectional (bool)    : set the LSTM do be bidirectional or not
    
    Return:
        output (torch.FloatTensor)  : logits over the words in the vocabulary (not normalized)
        hidden (tuple)              : tuple with hidden state and cell state
'''
class AT_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers=1, bidirectional=False):
        super(AT_LSTM, self).__init__()
        self.embedding_dim = embedding_size
        self.vocab_dim = vocab_size
        self.bidirectional = 2 if bidirectional else 1 
        self.num_layers = num_layers * self.bidirectional
        self.hidden_dim = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        
        self.attention = AttentionMechanism(hidden_size*self.bidirectional, dropout=0.5)
        self.output_layer = nn.Linear(hidden_size*self.bidirectional, vocab_size)

    def forward(self, input, hidden):
        input = input.permute(1, 0)
        input_emb = self.embedding(input)
        lstm_output, hidden = self.lstm(input_emb, hidden)

        attention_weights = self.attention(lstm_output)
        output = lstm_output * attention_weights
        output = torch.sum(output, dim=1)

        output = self.output_layer(output)

        return output, hidden

    def init_hidden(self, batch_size, use_cuda):
        '''
            Initializes weights to 0
        '''
        weights = next(self.parameters()).data
        if use_cuda:
            hidden = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                        weights.new(self.num_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weights.new(self.num_layers, batch_size, self.hidden_dim).zero_(), 
                        weights.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden

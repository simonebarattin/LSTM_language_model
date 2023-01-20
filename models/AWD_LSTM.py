import torch
import math
import torch.nn as nn

class WeightDrop(nn.Module):
    '''
        DropConnect on recurrent hidden to hidden weight matrices [1].
        The code used was implemented in https://github.com/salesforce/awd-lstm-lm

        References:
            [1] Merity et al. "Regularizing and Optimizing LSTM Language Models"
                ICLR 2018.
    '''
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
        self.module.flatten_parameters()
        return self.module.forward(*args)

class LockedDropout(nn.Module):
    '''
        Selects a unique dropout mask for different samples, which stays the same within the forward and backward pass [1].
        The code used was implemented in https://github.com/salesforce/awd-lstm-lm

        References:
            [1] Merity et al. "Regularizing and Optimizing LSTM Language Models"
                ICLR 2018.
    '''
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

class EmbeddDropout(nn.Module):
    '''
        Applies a dropout mask directly at embedding weight matrix [1]. 
        The code used was implemented in https://github.com/salesforce/awd-lstm-lm

        References:
            [1] Merity et al. "Regularizing and Optimizing LSTM Language Models"
                ICLR 2018.
    '''
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

class AWDLSTM(nn.Module):
    '''
        A script implementing the regularization techniques used in [1].

        Inputs:
            vocab_size      : size of the vocabulary (e.g. 10000 for Penn-Tree Bank dataset) used to create the word embeddings and the output of the
                              FC layer
            embedding_dim   : dimensionality of the embedding (defualt=400)
            hidden_dim      : dimensionality of the hidden layer output (default=1150)
            n_layers        : number of LSTM layers
            dropout         : probability of dropout at word level on the output before the FC layer
            dropoute        : probability of dropout on the embedding weight matrix
            dropoutw        : probability of dropout on the hidden-to-hidden weight matrix
            dropouti        : probability of dropout at word level on input to the LSTM model
            dropouth        : probability of dropout at word level between LSTM layers
            tweights        : tye weights

        References:
            [1] Merity et al. "Regularizing and Optimizing LSTM Language Models"
                ICLR 2018.
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, dropoute, dropoutw, dropouti, dropouth, tweights=True):
        super(AWDLSTM, self).__init__()
        self.vocab_size = vocab_size 
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.tweights = tweights

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
            if i != self.n_layers-1:
                out = self.lockeddrop(out, self.dropouth, self.training)
        out = self.lockeddrop(out, self.dropout, self.training)
        out = out.view(out.size(0)*out.size(1), out.size(2))
        out = self.fc(out)

        return out, new_h

    def init_hidden(self, batch_size, use_cuda):
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        if use_cuda:
            hidden = [(weights.new(1, batch_size, self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).zero_().cuda(), 
                        weights.new(1, batch_size, self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).zero_().cuda()) 
                        for i in range(self.n_layers)]
        else:
            hidden = [(weights.new(1, batch_size, self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).zero_(), 
                        weights.new(1, batch_size, self.hidden_dim if i!=self.n_layers-1 else (self.embedding_dim if self.tweights else self.hidden_dim)).zero_()) 
                        for i in range(self.n_layers)]
        
        return hidden
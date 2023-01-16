import torch

class PTBDataset():
    '''
        A script that creates a dataset structure for the Penn-Treebank dataset. Given the vocabulary, tokens and batch size
        creates a tensor with all the words indexes. Automatically drops the last incomplete batch.

        Reference:
            [1] https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/data_utils.py
            [2] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, vocab, tokens, batch_size) -> None:
        super(PTBDataset, self).__init__()
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

    def get_batch(self, idx, seq_len):
        '''
            Data shape returned torch.Size([sequence length, batch size])
                e.g. torch.Size([70, 64])
        '''
        x = self.data[idx:idx+seq_len]
        y = self.data[idx+1:idx+seq_len+1]
        return x, y
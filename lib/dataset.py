import torch

'''
        A script that creates a dataset structure for the Penn-Treebank dataset. Given the vocabulary, tokens and batch size
        creates a tensor with all the words indexes. Automatically drops the last incomplete batch.

        Args:
            vocab (Vocabulary)  : vocabulary of words in the dataset
            tokens (list)       : list of tokens
            batch_size (int)    : batch size
            mode (str)          : defines which model is in use

        Reference:
            [1] https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/data_utils.py
            [2] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
class PTBDataset():
    def __init__(self, vocab, tokens, batch_size, mode) -> None:
        super(PTBDataset, self).__init__()
        self.idx2word = vocab.idx2word
        self.word2idx = vocab.word2idx
        self.batch_size = batch_size
        self.tokens = tokens
        self.mode = mode
        self.data = torch.LongTensor(len(tokens))
        for count, t in enumerate(self.tokens):
            self.data[count] = self.word2idx[t]
        num_batches = self.data.size(0) // batch_size
        self.data = self.data[:num_batches*batch_size]
        self.data = self.data.view(batch_size, -1).t().contiguous()

    def get_batch(self, idx, seq_len):
        '''
            Data shape returned torch.Size([sequence length, batch size])
        '''
        x = self.data[idx:idx+seq_len]
        if self.mode == 'attention':
            try:
                y = self.data[idx+seq_len]
            except:
                y = self.data[-1]
        else:
            y = self.data[idx+1:idx+seq_len+1]
        return x, y
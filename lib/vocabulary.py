import torch
from collections import defaultdict

class Vocabulary():
    def __init__(self):
        self.num_tokens = 0
        self.word2idx = {}
        self.idx2word = {}
        self.words_ratio = {}
        self.frequency_list = defaultdict(int)

    def add2vocab(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_tokens
            self.idx2word[self.num_tokens] = word
            self.num_tokens += 1

    def process_tokens(self, tokens):
        word_ratio = []
        for token in tokens:
            self.add2vocab(token)
            self.frequency_list[token] += 1
        for key, val in self.idx2word.items():
            word_ratio.append(self.frequency_list[val] / len(self.idx2word))
        word_weigths = torch.FloatTensor(word_ratio)
        return word_weigths

    def __len__(self):
        return len(self.idx2word)
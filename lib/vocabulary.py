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
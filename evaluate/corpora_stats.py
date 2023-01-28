import sys
sys.path.append('.')
sys.path.append('..')

import os
from lib import *
from utils import load_data_tokenize

def corpora_stats():
    train_tokens = load_data_tokenize(DATASET['train'])
    valid_tokens = load_data_tokenize(DATASET['valid'])
    test_tokens = load_data_tokenize(DATASET['test'])

    # Check Out Of Vocabulary words
    print("#. Checking OOV words...")
    train_set = set(train_tokens)
    val_set = set(valid_tokens)
    test_set = set(test_tokens)
    print("  \\__Test-Train: ",len(test_set.difference(train_set)), " Val-Train: ", len(val_set.difference(train_set)))

    # Print top-10 most frequent words
    vocab = Vocabulary()
    vocab.add2vocab("<unk>")
    vocab.add2vocab("<eos>")
    vocab.process_tokens(train_tokens)
    
    print("\n#. Top-10 most frequent words")
    sorted_list = dict(sorted(vocab.frequency_list.items(), key=lambda item: item[1], reverse=True))
    for k in list(sorted_list.keys())[:10]:
        print("  \\__{}: {}".format(k, sorted_list[k]))

    print("\n#. Top-10 less frequent words")
    sorted_list = dict(sorted(vocab.frequency_list.items(), key=lambda item: item[1], reverse=False))
    for k in list(sorted_list.keys())[:10]:
        print("  \\__{}: {}".format(k, sorted_list[k]))


if __name__ == '__main__':
    corpora_stats()
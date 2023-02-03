import sys
sys.path.append('.')
sys.path.append('..')

from lib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

'''
    Script to extract information from the dataset corpora, e.g. words frequency, OOV words, ...
'''
def corpora_stats():
    
    def token_sentence_level(path):
        tokens = []
        sent_lengths = []
        with open(path, 'r') as f:
            corpus = f.readlines()
        for line in corpus:
            sent_lengths.append(len(line.split() + ['<eos>']))
            tokens.extend(line.split() + ['<eos>'])
        return sent_lengths, tokens

    train_sent_lengths, train_tokens = token_sentence_level(DATASET['train'])
    valid_sent_lengths, valid_tokens = token_sentence_level(DATASET['valid'])
    test_sent_lengths, test_tokens = token_sentence_level(DATASET['test'])
    print("#. Number of sentences | Max length | Avg Length")
    print("  \\_Train set: {} \t {} \t {}".format(len(train_sent_lengths), max(train_sent_lengths), int(sum(train_sent_lengths)/len(train_sent_lengths))))
    print("  \\_Validation set: {} \t {} \t {}".format(len(valid_sent_lengths), max(valid_sent_lengths), int(sum(valid_sent_lengths)/len(valid_sent_lengths))))
    print("  \\_Test set: {} \t {} \t {}".format(len(test_sent_lengths), max(test_sent_lengths), int(sum(test_sent_lengths)/len(test_sent_lengths))))
    print("#. Number of tokens:")
    print("  \\_Train set: {}".format(len(train_tokens)))
    print("  \\_Validation set: {}".format(len(valid_tokens)))
    print("  \\_Test set: {}".format(len(test_tokens)))

    # Check Out Of Vocabulary words
    print("#. Checking OOV words...")
    train_set = set(train_tokens)
    val_set = set(valid_tokens)
    test_set = set(test_tokens)
    print("  \\__Test-Train: ",len(test_set.difference(train_set)), " Val-Train: ", len(val_set.difference(train_set)))

    print("#. Number of words:")
    print("  \\_Train set: {}".format(len(train_set)))
    print("  \\_Validation set: {}".format(len(val_set)))
    print("  \\_Test set: {}".format(len(test_set)))

    # Print top-10 most/less frequent words
    vocab = Vocabulary()
    vocab.add2vocab("<unk>")
    vocab.add2vocab("<eos>")
    vocab.process_tokens(train_tokens)
    
    print("\n#. Top-10 most frequent words")
    rev_sorted_list = dict(sorted(vocab.frequency_list.items(), key=lambda item: item[1], reverse=True))
    max_count = len(train_tokens)
    for i, k in enumerate(list(rev_sorted_list.keys())[:10]):
        zipf_freq = rev_sorted_list[k] * (1 / (i+1))
        print("  \\__{}: {} \t Frequency: {:.4f} \tRank: 1/{} \t Zipf frequency: {:.2f}".format(k, rev_sorted_list[k], round(rev_sorted_list[k]/max_count, 4), i+1, zipf_freq))

    print("\n#. Top-10 less frequent words")
    for i, k in enumerate(list(rev_sorted_list.keys())[-10:]):
        i += len(list(rev_sorted_list.keys())[:-10])
        zipf_freq = rev_sorted_list[k] * (1 / (i+1))
        print("  \\__{}: {} \t Frequency: {:.4f} \tRank: 1/{} \t Zipf frequency: {:.2f}".format(k, rev_sorted_list[k], round(rev_sorted_list[k]/max_count, 4), i+1, zipf_freq))
    return rev_sorted_list

def plot_zipf(frequencies):    
    # Power law
    y = sorted(frequencies.values(), reverse=True)
    x = np.array(range(1, len(y)+1))
    plt.figure()
    plt.loglog()
    plt.plot(x, y, 'r+')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.savefig('zipf_power_law.png')
    plt.show()

    def smoothify(val):
        x = np.array(range(0, depth))
        y = np.array(val)
        x_smooth = np.linspace(x.min(), x.max(), 600)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth

    # Frequency law for the first 100 ranked words 
    depth = 100
    x_axs = list(range(0, depth, 10))

    words = list(frequencies.keys())[:depth]
    max_freq = frequencies[words[0]]
    y_axs = [round(frequencies[key] / max_freq * 100) for key in words]
    x, y = smoothify(y_axs)
    plt.plot(x, y, label="PTB Zipf's Curve", lw=2, alpha=0.5, color='red')

    plt.xticks(range(0, depth, 10), x_axs)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('zipf_frequency_law.png')
    plt.show()

if __name__ == '__main__':
    frequencies = corpora_stats()
    plot_zipf(frequencies)
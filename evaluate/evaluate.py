import sys
sys.path.append('.')
sys.path.append('..')

import torch
import argparse
import numpy as np
import torch.nn as nn
from lib import *
from models import *
from utils import load_data_tokenize, detach_hidden

'''
    Script to evaluate loss and perplexity of one of the pre-trained models on the dataset's test set.

    Usage example:
        python evaluate/evaluate.py --weight models/model_weights/vanilla-lstm.pth --cuda
'''
def evaluate():
    parser = argparse.ArgumentParser(description="Script to evaluate the trained models")
    parser.add_argument('--weight', default='models/model_weights/vanilla-lstm.pth', help="Path to the pth save file of the model to evaluate.")
    parser.add_argument('--cuda', action='store_true', help="Use GPU acceleration.")
    parser.add_argument('--wb', action='store_true', help="Use Weight&Biases to draw performances graphs.")
    args = parser.parse_args()

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################

    use_cuda = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    test_batch_size = TEST_BATCH_SIZE
    seq_len = SEQ_LEN_AWD_VANILLA

    embedding_size = EMBEDDING_SIZE
    hidden_size = HIDDEN_SIZE
    n_layers = NUM_LAYERS

    seq_len_att = SEQ_LEN_ATT

    seq_len_cnn = SEQ_LEN_CNN
    embedding_cnn = EMBEDDING_SIZE_CNN
    kernel_size = KERNEL_SIZE
    out_channels = OUT_CHANNELS
    num_layers_cnn = NUM_LAYERS_CNN
    bottleneck = BOTTLENECK

    dropout = 0.
    dropout_emb = 0.
    dropout_inp = 0.
    dropout_hid = 0.
    dropout_wgt = 0.

    train_tokens = load_data_tokenize(DATASET['train'])
    test_tokens = load_data_tokenize(DATASET['test'])

    vocab = Vocabulary()
    vocab.add2vocab("<unk>")
    vocab.add2vocab("<eos>")
    _ = vocab.process_tokens(train_tokens)

    weight_filename = args.weight.split('/')[-1]
    checkpoint = torch.load(args.weight, map_location=torch.device('cpu')) if not args.cuda else torch.load(args.weight)
    model_percs = weight_filename.split('.')[0].split('_')
    if model_percs[0] == 'vanilla-lstm':
        model = VanillaLSTM(len(vocab), embedding_size, embedding_size, num_layers=1)
        mode = 'vanilla'
    elif model_percs[0] == 'awd-lstm':
        if 'tyeweights' in model_percs:
            model = AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tweights=True)
        else:
            model = AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tweights=False)
        mode = 'awd'
    elif model_percs[0] == 'attention-lstm':
        seq_len = seq_len_att
        model = AT_LSTM(embedding_size, hidden_size, len(vocab), num_layers=1)
        mode = 'attention'
    else:
        seq_len = seq_len_cnn
        model = CNN_LM(len(vocab), embedding_cnn, kernel_size, out_channels, num_layers_cnn, bottleneck)
        mode = 'cnn'
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss()

    test_ids = PTBDataset(vocab, test_tokens, test_batch_size, mode)

    if args.wb:
        w_b = WeightBiases(dict(
            name=weight_filename.split('.')[0],
        ))

    print("\n#. {}".format(weight_filename.split('.')[0]))
    print("  \\__Testing...")
    hidden = model.init_hidden(test_batch_size, use_cuda) if not isinstance(model, CNN_LM) else None
    losses = []
    ppls = []
    model.eval()
    with torch.no_grad():
        for i in range(0, test_ids.data.size(0) - seq_len, seq_len):
            x, y = test_ids.get_batch(i, seq_len)
            x = x.cuda() if use_cuda else x
            y = y.cuda() if use_cuda else y

            if not isinstance(model, CNN_LM): h = detach_hidden(hidden)

            if not isinstance(model, CNN_LM):
                output, h = model(x, h)
                y = y.reshape(-1)
                loss = criterion(output, y)
            else:
                output, loss = model(x)

            cur_ppl = np.exp(loss.item())

            losses.append(loss.item())
            ppls.append(np.exp(loss.item()))

        cur_loss = np.mean(losses)
        cur_ppl = np.exp(cur_loss)
        
        if args.wb: w_b.log({"Test/Loss": cur_loss, "Test/PPL": cur_ppl})

        print("    \\__Loss: {}".format(cur_loss))
        print("    \\__PPL: {}".format(cur_ppl))

    if args.wb: w_b.finish()

if __name__ == '__main__':
    evaluate()
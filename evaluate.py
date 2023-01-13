import torch
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
from lib import *
from models import *
from .utils import load_data_tokenize, detach_hidden

def evaluate():
    '''
        A script to evaluate loss and perplexity of pre-trained model.

    '''
    parser = argparse.ArgumentParser(description="Script to evaluate the trained models")
    parser.add_argument('--weight', default='vanilla-lstm.pth', help="Name of the pth save file of the model to evaluate.")
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

    test_batch_size = 10
    seq_len = 70

    embedding_size = 400
    hidden_size = 1150
    n_layers = 3

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
    vocab.process_tokens(train_tokens)

    test_ids = PTBDataset(vocab, test_tokens, test_batch_size)

    weight_filename = args.weight
    checkpoint = torch.load(osp(BEST_MODEL_PATH, weight_filename))
    model_percs = weight_filename.split('.')[0].split('_')
    if model_percs[0] == "vanilla-lstm":
        model = VanillaLSTM(len(vocab), embedding_size, hidden_size, n_layers, test_batch_size)
    else:
        if 'tyeweights' in model_percs:
            model = AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tye_weights=True)
        else:
            model = AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tye_weights=False)
    model = model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss()

    if args.wb:
        w_b = WeightBiases(dict(
            name=weight_filename.split('.')[0],
        ))

    print("\n#. {}".format(weight_filename.split('.')[0]))
    print("  \\__Testing...")
    hidden = model.init_hidden(test_batch_size)
    losses = []
    ppls = []
    model.eval()
    with torch.no_grad():
        for i in range(0, test_ids.data.size(0) - seq_len, seq_len):
            x, y = test_ids.get_batch(i, seq_len)
            x = x.cuda() if use_cuda else x
            y = y.cuda() if use_cuda else y
            # h = tuple([each.data for each in hidden])
            h = detach_hidden(hidden)

            output, h = model(x, h)
            # output = output.reshape(test_batch_size * seq_len, -1)
            y = y.reshape(-1)

            loss = criterion(output, y)
            cur_ppl = np.exp(loss.item())

            losses.append(loss.item())
            ppls.append(np.exp(loss.item()))

        cur_loss = np.mean(losses)
        cur_ppl = np.exp(cur_loss)
        
        w_b.log({"Test/Loss": cur_loss, "Tets/PPL": cur_ppl})

        print("    \\__Loss: {}".format(cur_loss))
        print("    \\__PPL: {}".format(cur_ppl))

    w_b.finish()

if __name__ == '__main__':
    evaluate()
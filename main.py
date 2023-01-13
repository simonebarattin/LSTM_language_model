import torch
import argparse
import os.path as osp
import torch.nn as nn
from lib import *
from models import *
from trainer import train, valid
from utils import load_data_tokenize, save_model, concat_name

def main():
    '''
        Main script for the Language Modeling project. Trains a model to predict the probability of a word 
        in a vocabulary given the previous n words. The models defined are Vanilla LSTM (baseline), AWD-LSTM [1], and 
        LSTM with an attention layer.
        The model parameters used are the ones provided in [1].

        References:
            [1] Merity et al. "Regularizing and Optimizing LSTM Language Models"
                ICLR 2018.
    '''
    parser = argparse.ArgumentParser(description="Main script to run the project.")
    parser.add_argument('--baseline', action='store_true', help="Train a vanilla LSTM baseline.")
    parser.add_argument('--awd', action='store_true', help="Train a AWD LSTM.")
    parser.add_argument('--attention', action='store_true', help="Train a LSTM with attention.")
    parser.add_argument('--asgd', action='store_true', help="Use the Averaged SGD (after SGD converges).")
    parser.add_argument('--dropout', type=float, default=0.5, help="Apply dropout.")
    parser.add_argument('--dropout-emb', type=float, default=0.0, help="Apply dropout to the embeddings' matrices.")
    parser.add_argument('--dropout-inp', type=float, default=0.0, help="Apply dropout on the input's recurrent connection.")
    parser.add_argument('--dropout-hid', type=float, default=0.0, help="Apply dropout on the hidden's recurrent connection.")
    parser.add_argument('--dropout-wgt', type=float, default=0.0, help="Apply dropout on the weights matrices.")
    parser.add_argument('--clip-gradient', action='store_true', help="Use clip gradient regularization.")
    parser.add_argument('--tye-weights', action='store_true', help="Use regualrization tye weights.")
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
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                  [ PARAMETERS ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################

    if args.clip_gradient:
        clip_gradient = 0.25
    tye_weights = args.tye_weights
    asgd = args.asgd

    dropout = args.dropout
    dropout_emb = args.dropout_emb
    dropout_inp = args.dropout_inp
    dropout_hid = args.dropout_hid
    dropout_wgt = args.dropout_wgt

    embedding_size = 400
    hidden_size = 1150
    n_layers = 3

    epochs = 100
    train_batch_size = 20
    seq_len = 70
    seq_len_threshold = 0.8

    lr = 30
    weight_decay = 1e-6
    patience = 5

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                               [ PREPROCESSING ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################

    train_tokens = load_data_tokenize(DATASET['train'])
    valid_tokens = load_data_tokenize(DATASET['valid'])
    test_tokens = load_data_tokenize(DATASET['test'])

    print("#. Checking OOV words...")
    train_set = set(train_tokens)
    val_set = set(valid_tokens)
    test_set = set(test_tokens)
    print("  \\__Test-Train: ",len(test_set.difference(train_set)), " Val-Train: ", len(val_set.difference(train_set)))

    vocab = Vocabulary()
    vocab.add2vocab("<unk>")
    vocab.add2vocab("<eos>")
    vocab.process_tokens(train_tokens)

    # TODO find some corpora stats: word frequency with graph, sentence length, ...

    train_ids = PTBDataset(vocab, train_tokens, train_batch_size)
    valid_ids = PTBDataset(vocab, valid_tokens, train_batch_size)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                      [ MODELS / OPTIMIZER / CRITERION ]                                        ##
    ##                                                                                                                ##
    ####################################################################################################################

    models = []
    if args.baseline:
        models.append(("vanilla-lstm", VanillaLSTM(len(vocab), embedding_size, hidden_size, n_layers)))
    if args.awd:
        name = concat_name("awd-lstm",  asgd, args.clip_gradient, tye_weights, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid)
        models.append((name, AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tye_weights)))
    if args.attention:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                          [ TRAINING / VALIDATION]                                              ##
    ##                                                                                                                ##
    ####################################################################################################################

    for (name, model) in models:
        print("\n#. {}".format(name.split('_')[0]))
        if name.startswith("awd-lstm"):
            print("  \\__Averaged SGD: {}".format(asgd))
            print("  \\__Clip gradient: {}".format(clip_gradient if args.clip_gradient else False))
            print("  \\__Tye Weights: {}".format(tye_weights))
            print("  \\__Dropout: {}".format(dropout))
            print("  \\__Dropout embedding: {}".format(dropout_emb))
            print("  \\__Dropout input: {}".format(dropout_inp))
            print("  \\__Dropout hidden: {}".format(dropout_hid))
            print("  \\__Dropout weights: {}".format(dropout_wgt))

        model = model.cuda() if use_cuda else model
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.wb:
            w_b = WeightBiases(dict(
                name=name,
            ))
        else:
            w_b = None

        best_ppl = float('inf')
        best_loss = float('inf')

        for epoch in range(epochs):
            print("\n#. Epoch {}".format(epoch))
            print("  \\__Training...")
            train_loss, train_ppl = train(train_ids, model, optimizer, criterion, lr, train_batch_size, seq_len, seq_len_threshold, w_b, use_cuda, clip_gradient if args.clip_gradient else None, epoch)
            print("    \\__Loss: {}".format(train_loss))
            print("    \\__PPL: {}".format(train_ppl))

            if args.wb: w_b.log({"Training/Average Loss": train_loss, "Training/Average PPL": train_ppl})

            print("  \\__Validation...")
            valid_loss, valid_ppl = valid(valid_ids, model, criterion, train_batch_size, seq_len, w_b, use_cuda, epoch)
            print("    \\__Loss: {}".format(train_loss))
            print("    \\__PPL: {}".format(train_ppl))

            if args.wb: w_b.log({"Validation/Average Loss": valid_loss, "Validation/Average PPL": valid_ppl})

            if valid_ppl < best_ppl:
                print("      \\__Save model at epoch {} with best perplexity {}".format(epoch, valid_ppl))
                save_model(model, osp.join(BEST_MODEL_PATH, "{}.pth".format(name)))
                best_ppl = valid_ppl
            if valid_loss >= best_loss:
                patience -= 1
                best_loss = valid_loss
                print("      \\__Loss not decreasing... Patience to {}".format(patience))
                if not asgd:
                    asgd = True
                    optimizer = torch.optim.ASGD(model.parameters(), lr=lr, lambd=0., weight_decay=weight_decay, t0=0)
                    print("      \\__Switching to ASGD with lr {}".format(lr))
                if patience == 0:
                    print("      \\__Training stopped due to early stopping!")
                    break
            else:
                best_loss = valid_loss
                patience = 5
        if args.wb:
            w_b.finish()
        

if __name__ == '__main__':
    main()
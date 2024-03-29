import sys
sys.path.append('.')
sys.path.append('..')

import torch
import argparse
from lib import *
from models import *
from utils import load_data_tokenize

'''
    Script that, given a pre-trained model, tests its ability to generate text given a prompt (in this case 'the' it's the prompt). The generation process uses
    different temperatures (check [1] for a simple explanation on temperature). The generation continues unitl the '<eos>' token is generated or the maximum sequence length is reached.

    Reference:
        [1] https://lukesalamone.github.io/posts/what-is-temperature/

    Args:
        prompt (str)        : starting point of the generation
        max_seq_len (int)   : maximum length of the generated sequence
        temperature (float) : temperature value
        model (nn.Module)   : pre-trained model
        vocab (Vocabulary)  : vocabulary of words
        use_cuda (bool)     : use CUDA or not
        seed (int)          : set generation seed

    Returns:
        tokens (list)       : list of tokens
'''
def generate(prompt, max_seq_len, temperature, model, vocab, use_cuda, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = prompt.split()
    indices = [vocab.word2idx[t] for t in tokens]
    inp = torch.LongTensor([indices]).cuda() if use_cuda else torch.LongTensor([indices])
    batch_size = 1
    hidden = model.init_hidden(batch_size, use_cuda)
    with torch.no_grad():
        for i in range(max_seq_len):
            prediction, hidden = model(inp, hidden)
            probs = prediction.squeeze().data.div(temperature).exp()
            prediction = torch.multinomial(probs, num_samples=1).item()
            
            while prediction == vocab.word2idx['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab.word2idx['<eos>']:
                break
            
            indices.append(prediction)
            inp.data.fill_(prediction)

    tokens = [vocab.idx2word[i] for i in indices]
    return tokens

'''
    Main script to test the text generation abilities of a pre-trained model.

    Usage example:
        python evaluate/inference.py --weight models/model_weights/vanilla-lstm.pth --cuda

'''
def main():
    parser = argparse.ArgumentParser(description="Script to use a pre-trained model to generate text given a prompt")
    parser.add_argument('--weight', default='vanilla-lstm.pth', help="Path to the pth save file of the model to use.")
    parser.add_argument('--cuda', action='store_true', help="Use GPU acceleration.")
    args = parser.parse_args()

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

    embedding_size = 400
    hidden_size = 1150
    n_layers = 3

    embedding_cnn = 200
    kernel_size = 2
    out_channels = 600
    num_layers_cnn = 4
    bottleneck = 20

    dropout = 0.
    dropout_emb = 0.
    dropout_inp = 0.
    dropout_hid = 0.
    dropout_wgt = 0.

    train_tokens = load_data_tokenize(DATASET['train'])

    vocab = Vocabulary()
    vocab.add2vocab("<unk>")
    vocab.add2vocab("<eos>")
    _ = vocab.process_tokens(train_tokens)

    weight_filename = args.weight.split('/')[-1]
    checkpoint = torch.load(args.weight, map_location=torch.device('cpu')) if not args.cuda else torch.load(args.weight)
    model_percs = weight_filename.split('.')[0].split('_')
    if model_percs[0] == 'vanilla-lstm':
        model = VanillaLSTM(len(vocab), embedding_size, embedding_size, num_layers=1)
        # mode = 'vanilla'
    elif model_percs[0] == 'awd-lstm':
        if 'tyeweights' in model_percs:
            model = AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tweights=True)
        else:
            model = AWDLSTM(len(vocab), embedding_size, hidden_size, n_layers, dropout, dropout_emb, dropout_wgt, 
                                            dropout_inp, dropout_hid, tweights=False)
        # mode = 'awd'
    elif model_percs[0] == 'attention-lstm':
        model = AT_LSTM(embedding_size, hidden_size, len(vocab), num_layers=1)
        # mode = 'attention'
    else:
        model = CNN_LM(len(vocab), embedding_cnn, kernel_size, out_channels, num_layers_cnn, bottleneck)
        # mode = 'cnn'
    model.load_state_dict(checkpoint['state_dict'])

    prompt = 'the'
    max_seq_len = 30
    seed = 0

    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, vocab, use_cuda, seed)
        print("#.Temperature {}".format(temperature))
        print("  \\__generated: {}".format(generation))

if __name__ == '__main__':
    main()
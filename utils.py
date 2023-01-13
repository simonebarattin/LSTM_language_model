import torch

def load_data_tokenize(path):
    '''
        A script for loading from a corpora from a text file and tokenize it. It adds automatically the end-of-sentence special token <eos>
        at the end of the sentence.
    '''
    tokens = []
    with open(path, 'r') as f:
        corpus = f.readlines()
    for line in corpus:
        tokens.extend(line.split() + ['<eos>'])
    return tokens

def save_model(model, path):
    torch.save({
        "state_dict": model.state_dict()
    }, path)

def detach_hidden(hidden):
    if len(hidden) > 0:
        detached = []
        for h in hidden:
            detached.append(tuple([h[0].detach(), h[1].detach()]))
    else:
        detached = tuple([h[0].detach(), h[1].detach()])
    return detached

def concat_name(model, asgd, clip_gradient, tye_weights, dropout, dropout_emb, dropout_wgt, dropout_inp, dropout_hid):
    name = model
    if asgd:
        name += "_asgd"
    if clip_gradient:
        name += "_clipgradient"
    if tye_weights:
        name += "_tyeweights"
    if dropout != 0.0:
        name += "_dropout"
    if dropout_emb != 0.0:
        name += "_dropoutemb"
    if dropout_hid != 0.0:
        name += "_dropouthid"
    if dropout_wgt != 0.0:
        name += "_dropoutwgt"
    if dropout_inp != 0.0:
        name += "_dropoutinp"
    return name
import torch
import random
import numpy as np
import torch.nn as nn
from utils import detach_hidden
from models import *
from sklearn.metrics import f1_score

def train(vocab, mode, train_ids, model, optimizer, criterion, lr, batch_size, seq_len, seq_len_threshold, w_b, use_cuda, clip_grad, epoch):
    hidden = model.init_hidden(batch_size, use_cuda) if mode != 'cnn' else None
    losses = []
    ppls = []
    # f1s = []

    # batch, i = 0, 0

    mu = seq_len if random.random() < seq_len_threshold else seq_len/2
    std = 5
    model.train()
    # use while since bptt changes at each step
    while i < train_ids.data.size(0) -1 -1:
        # following the paper, sample different sequence lengths in order to learn throughout all the corpus
        if isinstance(model, AWDLSTM):
            bptt = max(std, int(np.random.normal(mu, std)))
            new_lr = (lr * bptt) / mu
            optimizer.param_groups[0]['lr'] = new_lr
        else:
            bptt = seq_len

        x, y = train_ids.get_batch(i, bptt)
        if x.shape[1] != y.shape[-1]:
            break

        x = x.cuda() if use_cuda else x
        y = y.cuda() if use_cuda else y
        if mode != 'cnn': h = detach_hidden(hidden)

        optimizer.zero_grad()
        if mode != 'cnn':
            output, h = model(x, h)
            y = y.reshape(-1)
            
            # f1 = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='micro')
            # f1s.append(f1)

            loss = criterion(output, y)
        else:
            output, loss = model(x)
            # f1 = 0.

        loss.backward()
        if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), clip_grad) 
        optimizer.step()
        
        losses.append(loss.item())
        cur_ppl = np.exp(loss.item())

        # if w_b is not None:
        #     w_b.log({"Training/Loss": loss.item(), "Training/PPL": cur_ppl, "Training/F1": f1})
        #     w_b.step_increment(1)

        ppls.append(cur_ppl)
        i += bptt

    cur_loss = np.mean(losses)
    cur_ppl = np.exp(cur_loss)
    # cur_f1 = np.mean(f1s)
    return cur_loss, cur_ppl#, cur_f1

def valid(vocab, mode, valid_ids, model, criterion, batch_size, seq_len, w_b, use_cuda, epoch):
    hidden = model.init_hidden(batch_size, use_cuda) if mode != 'cnn' else None
    losses = []
    ppls = []
    # f1s = []

    model.eval()
    with torch.no_grad():
        for i in range(0, valid_ids.data.size(0) - 1, seq_len):
            x, y = valid_ids.get_batch(i, seq_len)
            if x.shape[1] != y.shape[-1]:
                break

            x = x.cuda() if use_cuda else x
            y = y.cuda() if use_cuda else y
            if mode != 'cnn': h = detach_hidden(hidden)

            if mode != 'cnn':
                output, h = model(x, h)
                y = y.reshape(-1)
                preds = torch.argmax(torch.nn.Softmax(dim=1)(output), dim=1)
                # for j,_ in enumerate(y):
                #     print(vocab.idx2word[preds[j].item()], vocab.idx2word[y[j].item()])
                # preds = torch.argmax(output, dim=1)

                # f1 = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='micro')
                # f1s.append(f1)

                loss = criterion(output, y)
            else:
                output, loss = model(x)
                # f1 = 0.
            
            cur_ppl = np.exp(loss.item())

            # if w_b is not None:
            #     w_b.log({"Validation/Loss": loss.item(), "Validation/PPL": cur_ppl, "Validation/F1": f1})
            #     w_b.step_increment(1)

            losses.append(loss.item())
            ppls.append(np.exp(loss.item()))

        cur_loss = np.mean(losses)
        cur_ppl = np.exp(cur_loss)
        # cur_f1 = np.mean(f1s)
        return cur_loss, cur_ppl#, cur_f1
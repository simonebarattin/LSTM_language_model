import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import detach_hidden
from models import *
from lib import *

def train(mode, train_ids, model, optimizer, criterion, lr, batch_size, seq_len, seq_len_threshold, use_cuda, clip_grad, teacher_forcing=True):
    hidden = model.init_hidden(batch_size, use_cuda) if mode != 'cnn' else None
    losses = []
    ppls = []

    i = 0
    preds = None

    mu = seq_len if random.random() < seq_len_threshold else seq_len/2
    std = 5
    model.train()
    # use while since bptt changes at each step
    while i < train_ids.data.size(0):

        # following the paper, sample different sequence lengths in order to learn throughout all the corpus
        if isinstance(model, AWDLSTM):
            bptt = max(std, int(np.random.normal(mu, std)))
            new_lr = (lr * bptt) / mu
            optimizer.param_groups[0]['lr'] = new_lr
        else:
            bptt = seq_len

        x, y = train_ids.get_batch(i, bptt)
        if mode == 'attention':
            if x.shape[1] != y.shape[-1]:
                break
        else:
            if x.shape != y.shape:
                break

        x = x.cuda() if use_cuda else x
        y = y.cuda() if use_cuda else y
        if mode != 'cnn': h = detach_hidden(hidden)

        optimizer.zero_grad()
        if mode == 'cnn':
            output, loss = model(x)
        elif mode == 'attention':
            if i == 0:
                inp = x
            output, h = model(inp, h)
            loss = criterion(output, y)
            
            preds = torch.argmax(torch.nn.Softmax(dim=1)(output), dim=1)
            if teacher_forcing:
                if random.random() < TEACHER_FORCING_P: # probability of using ground truth or model prediction as next word
                    inp = torch.cat((inp[1:, :], preds.unsqueeze(0)), dim=0)
                else:
                    inp = torch.cat((inp[1:, :], y.unsqueeze(0)), dim=0)
        else:
            output, h = model(x, h)
            y = y.reshape(-1)
            loss = criterion(output, y)
        loss.backward()
        if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), clip_grad) 
        optimizer.step()
        
        losses.append(loss.item())
        cur_ppl = np.exp(loss.item())

        ppls.append(cur_ppl)
        i += bptt if mode != 'attention' else 1

    cur_loss = np.mean(losses)
    cur_ppl = np.exp(cur_loss)

    return cur_loss, cur_ppl

def valid(mode, valid_ids, model, criterion, batch_size, seq_len, use_cuda):
    hidden = model.init_hidden(batch_size, use_cuda) if mode != 'cnn' else None
    losses = []
    ppls = []

    model.eval()
    with torch.no_grad():
        for i in range(0, valid_ids.data.size(0) - 1, seq_len):
            x, y = valid_ids.get_batch(i, seq_len)
            if mode == 'attention':
                if x.shape[1] != y.shape[-1]:
                    break
            else:
                if x.shape != y.shape:
                    break

            x = x.cuda() if use_cuda else x
            y = y.cuda() if use_cuda else y
            if mode != 'cnn': h = detach_hidden(hidden)

            if mode != 'cnn':
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
        return cur_loss, cur_ppl
import torch
import random
import numpy as np
import torch.nn as nn
from utils import detach_hidden

def train(train_ids, model, optimizer, criterion, lr, batch_size, seq_len, seq_len_threshold, w_b, use_cuda, clip_grad, epoch):
    hidden = model.init_hidden(batch_size, use_cuda)
    losses = []
    ppls = []

    batch, i = 0, 0

    mu = seq_len if random.random() < seq_len_threshold else seq_len/2
    std = 5
    model.train()
    # use while since bptt changes at each step
    while i < train_ids.data.size(0) -1 -1:
        # following the paper, sample different sequence lengths in order to learn throughout all the corpus
        bptt = max(std, int(np.random.normal(mu, std)))
        new_lr = (lr * bptt) / mu
        optimizer.param_groups[0]['lr'] = new_lr

        x, y = train_ids.get_batch(i, bptt)
        if x.shape!=y.shape:
            break
        x = x.cuda() if use_cuda else x
        y = y.cuda() if use_cuda else y
        h = detach_hidden(hidden)

        optimizer.zero_grad()
        output, h = model(x, h)

        # output = output.reshape(batch_size * seq_len, -1)
        y = y.reshape(-1)

        loss = criterion(output, y)
        loss.backward()
        if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), clip_grad) 
        optimizer.step()
        
        losses.append(loss.item())
        cur_ppl = np.exp(loss.item())

        if w_b is not None:
            w_b.log({"Training/Loss": loss.item(), "Training/PPL": cur_ppl})
            w_b.step_increment(1)

        ppls.append(cur_ppl)
        i += bptt

    cur_loss = np.mean(losses)
    cur_ppl = np.exp(cur_loss)
    return cur_loss, cur_ppl

def valid(valid_ids, model, criterion, batch_size, seq_len, w_b, use_cuda, epoch):
    hidden = model.init_hidden(batch_size, use_cuda)
    losses = []
    ppls = []
    model.eval()
    with torch.no_grad():
        # for i in tqdm(range(0, val_ids.data.size(1) - seq_len, seq_len)):
        for i in range(0, valid_ids.data.size(0) - 1, seq_len):
            x, y = valid_ids.get_batch(i, seq_len)
            if x.shape!=y.shape:
                continue
            x = x.cuda() if use_cuda else x
            y = y.cuda() if use_cuda else y
            h = detach_hidden(hidden)
            # h = tuple([each.data for each in hidden])

            output, h = model(x, h)
            # output = output.reshape(batch_size * seq_len, -1)
            y = y.reshape(-1)

            loss = criterion(output, y)
            cur_ppl = np.exp(loss.item())

            if w_b is not None:
                w_b.log({"Validation/Loss": loss.item(), "Validation/PPL": cur_ppl})
                w_b.step_increment(1)

            losses.append(loss.item())
            ppls.append(np.exp(loss.item()))

        cur_loss = np.mean(losses)
        cur_ppl = np.exp(cur_loss)
        return cur_loss, cur_ppl
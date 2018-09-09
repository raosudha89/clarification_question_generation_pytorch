from constants import *
import torch
from torch.nn import functional
from torch.autograd import Variable


def calculate_log_probs(logits, output, length, loss_fn, mixer_delta):
    batch_size = logits.shape[0]
    log_probs = functional.log_softmax(logits, dim=2)
    avg_log_probs = Variable(torch.zeros(batch_size))
    if USE_CUDA:
        avg_log_probs = avg_log_probs.cuda()
    for b in range(batch_size):
        curr_len = length[b]-mixer_delta
        if curr_len > 0:
            avg_log_probs[b] = loss_fn(log_probs[b][:curr_len], output[b][:curr_len]) / curr_len
    return avg_log_probs


def masked_cross_entropy(logits, target, length, loss_fn, mixer_delta=None):
    batch_size = logits.shape[0]
    # log_probs: (batch, max_len, num_classes)
    log_probs = functional.log_softmax(logits, dim=2) 
    loss = 0.
    for b in range(batch_size):
        curr_len = min(length[b], mixer_delta)
        sent_loss = loss_fn(log_probs[b][:curr_len], target[b][:curr_len]) / curr_len
        loss += sent_loss
    loss = loss / batch_size
    return loss

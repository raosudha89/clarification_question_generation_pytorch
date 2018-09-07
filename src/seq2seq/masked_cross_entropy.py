from constants import *
import torch
from torch.nn import functional
from torch.autograd import Variable


def calculate_log_probs(logits, output, length, loss_fn):
    max_len = logits.shape[1]
    batch_size = logits.shape[0]
    log_probs = functional.log_softmax(logits, dim=2)
    avg_log_probs = Variable(torch.zeros(batch_size))
    if USE_CUDA:
        avg_log_probs = avg_log_probs.cuda()
    for b in range(batch_size):
        # avg_log_probs[b] = loss_fn(log_probs[b][mixer_delta:], output[b][mixer_delta:]) / (max_len - mixer_delta)
        if log_probs[b].shape[0] and length[b] > log_probs[b].shape[0]:
            avg_log_probs[b] = loss_fn(log_probs[b], output[b]) / (length[b] - log_probs[b].shape[0])
    return avg_log_probs


def masked_cross_entropy(logits, target, length, loss_fn, mixer_delta=None):
    batch_size = logits.shape[0]
    # log_probs: (batch, max_len, num_classes)
    log_probs = functional.log_softmax(logits, dim=2) 
    loss = 0.
    for b in range(batch_size):
        # sent_loss = loss_fn(log_probs[b][:mixer_delta], target[b][:mixer_delta]) / mixer_delta
        sent_loss = loss_fn(log_probs[b][:min(mixer_delta, length[b])],
                            target[b][:min(mixer_delta, length[b])]) / min(mixer_delta, length[b])
        loss += sent_loss
    loss = loss / batch_size
    return loss

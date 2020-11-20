from constants import *
import math
import numpy as np
import nltk
import random
import time
import torch


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def iterate_minibatches(id_seqs, input_seqs, input_lens, output_seqs, output_lens, batch_size, shuffle=True):
    if shuffle:
        indices = np.arange(len(input_seqs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(input_seqs) - batch_size + 1, batch_size):
        if shuffle:
            ex = indices[start_idx:start_idx + batch_size]
        else:
            ex = slice(start_idx, start_idx + batch_size)
        yield np.array(id_seqs)[ex], np.array(input_seqs)[ex], np.array(input_lens)[ex], \
                np.array(output_seqs)[ex], np.array(output_lens)[ex]


def reverse_dict(word2index):
    index2word = {}
    for w in word2index:
        ix = word2index[w]
        index2word[ix] = w
    return index2word


def calculate_bleu(true, true_lens, pred, pred_lens, index2word, max_len):
    sent_bleu_scores = torch.zeros(len(pred))
    for i in range(len(pred)):
        true_sent = [index2word[idx] for idx in true[i][:true_lens[i]]]
        pred_sent = [index2word[idx] for idx in pred[i][:pred_lens[i]]]
        sent_bleu_scores[i] = nltk.translate.bleu_score.sentence_bleu([true_sent], pred_sent)
    if USE_CUDA:
        sent_bleu_scores = sent_bleu_scores.cuda()
    return sent_bleu_scores

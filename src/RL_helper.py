from constants import *
import math
import numpy as np
import nltk
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

def iterate_minibatches(p, pl, q, ql, pq, pql, a, al, batch_size, shuffle=True):
	if shuffle:
		indices = np.arange(len(p))
		np.random.shuffle(indices)
	for start_idx in range(0, len(p) - batch_size + 1, batch_size):
		if shuffle:
			ex = indices[start_idx:start_idx + batch_size]
		else:
			ex = slice(start_idx, start_idx + batch_size)
		yield np.array(p)[ex], np.array(pl)[ex], np.array(q)[ex], np.array(ql)[ex], \
				np.array(pq)[ex], np.array(pql)[ex], np.array(a)[ex], np.array(al)[ex] 

def reverse_dict(word2index):
	index2word = {}
	for w, ix in word2index.iteritems():
		index2word[ix] = w
	return index2word

def calculate_bleu(true, true_lens, pred, pred_lens, index2word):
	bleu_scores = [None]*len(pred)
	for i in range(len(pred)):
		true_sent = [index2word[idx] for idx in true[i][:true_lens[i]]]
		pred_sent = [index2word[idx] for idx in pred[i][:pred_lens[i]]]
		bleu_scores[i] = nltk.translate.bleu_score.sentence_bleu(true_sent, pred_sent)
		#brevity_penalty = nltk.translate.bleu_score.brevity_penalty(len(true_sent), len(pred_sent))
		#bleu_scores[i] = bleu_scores[i] * brevity_penalty
		#bleu_scores[i] = len(list(set(true_sent).intersection(pred_sent)))*1.0/len(true_sent)
	bleu_scores = torch.FloatTensor(np.array(bleu_scores))
	if USE_CUDA:
		bleu_scores = bleu_scores.cuda()
	return bleu_scores

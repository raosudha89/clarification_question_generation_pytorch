import math
import numpy as np
import random
import time

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

def iterate_minibatches(input_seqs, input_lens, output_seqs, output_lens, batch_size, shuffle=True):
	if shuffle:
		indices = np.arange(len(input_seqs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(input_seqs) - batch_size + 1, batch_size):
		if shuffle:
			ex = indices[start_idx:start_idx + batch_size]
		else:
			ex = slice(start_idx, start_idx + batch_size)
		yield np.array(input_seqs)[ex], np.array(input_lens)[ex], \
			  np.array(output_seqs)[ex], np.array(output_lens)[ex]

def reverse_dict(word2index):
	index2word = {}
	for w, ix in word2index.iteritems():
		index2word[ix] = w
	return index2word


import math
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
		combined = list(zip(input_seqs, input_lens, output_seqs, output_lens))
		random.shuffle(combined)
		input_seqs, input_lens, output_seqs, output_lens = zip(*combined)
	for start_idx in range(0, len(input_seqs) - batch_size + 1, batch_size):
		excerpt = indices[start_idx:start_idx + batch_size]
		yield input_seqs[excerpt], input_lens[excerpt], output_seqs[excerpt], output_lens[excerpt] 

def reverse_dict(word2index):
	index2word = {}
	for w, ix in word2index.iteritems():
		index2word[ix] = w
	return index2word


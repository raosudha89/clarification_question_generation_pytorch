import numpy as np

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


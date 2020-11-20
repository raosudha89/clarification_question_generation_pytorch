import numpy as np
import torch
import torch.nn.functional as F


def iterate_minibatches(c, cm, q, qm, a, am, l, batch_size, shuffle=True):
	if shuffle:
		indices = np.arange(len(c))
		np.random.shuffle(indices)
	for start_idx in range(0, len(c) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield c[excerpt], cm[excerpt], q[excerpt], qm[excerpt], a[excerpt], am[excerpt], l[excerpt]


def get_masks(lens, max_len):
	masks = []
	for i in range(len(lens)):
		if lens[i] == None:
			lens[i] = 0
		masks.append([1]*int(lens[i])+[0]*int(max_len-lens[i]))
	return np.array(masks)


def binary_accuracy(predictions, truth):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""
	correct = 0.
	for i in range(len(predictions)):
		if predictions[i] >= 0.5 and truth[i] == 1:
			correct += 1
		elif predictions[i] < 0.5 and truth[i] == 0:
			correct += 1
	acc = correct/len(predictions)
	return acc


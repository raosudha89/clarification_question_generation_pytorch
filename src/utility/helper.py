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
		masks.append([1]*lens[i]+[0]*(max_len-lens[i]))
	return np.array(masks)


def binary_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""

	#round predictions to the closest integer
	rounded_preds = torch.round(F.sigmoid(preds))
	correct = (rounded_preds == y).float() #convert into float for division 
	acc = correct.sum()/len(correct)
	#pred_pos = (rounded_preds == 1).float()
	#y_pos = (y == 1).float()
	#acc = pred_pos.sum()/y_pos.sum()
	return acc


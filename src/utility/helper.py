import numpy as np
import torch
import torch.nn.functional as F

def iterate_minibatches(contexts, questions, answers, labels, batch_size, shuffle=True):
	if shuffle:
		indices = np.arange(len(contexts))
		np.random.shuffle(indices)
	for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield contexts[excerpt], questions[excerpt], answers[excerpt], labels[excerpt]

import torch.nn.functional as F

def binary_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""

	#round predictions to the closest integer
	rounded_preds = torch.round(F.sigmoid(preds))
	correct = (rounded_preds == y).float() #convert into float for division 
	acc = correct.sum()/len(correct)
	return acc


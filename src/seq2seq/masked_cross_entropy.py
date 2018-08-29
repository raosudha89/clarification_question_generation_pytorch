from constants import *
import numpy as np
import torch
from torch.nn import functional
from torch.autograd import Variable

def sequence_mask(sequence_length, max_len=None):
	if max_len is None:
		max_len = sequence_length.data.max()
	batch_size = sequence_length.size(0)
	seq_range = torch.arange(0, max_len).long()
	seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
	seq_range_expand = Variable(seq_range_expand)
	if USE_CUDA:
		if sequence_length.is_cuda:
			seq_range_expand = seq_range_expand.cuda()
	seq_length_expand = (sequence_length.unsqueeze(1)
						 .expand_as(seq_range_expand))
	return seq_range_expand < seq_length_expand

def calculate_log_probs(logits, masks):
	# log_probs: (batch * max_len * num_classes)
	log_probs = functional.log_softmax(logits, dim=2)
	# max_log_probs: (batch * max_len)
	max_log_probs = torch.topk(log_probs, 1, dim=2)[0][:,:,0]
	masks = torch.FloatTensor(masks)
	if USE_CUDA:
		masks = masks.cuda()
	max_log_probs = max_log_probs * masks
	#return max_log_probs
	avg_log_probs = Variable(torch.zeros(max_log_probs.shape[0]))
	if USE_CUDA:
		avg_log_probs = avg_log_probs.cuda()
	for i in range(max_log_probs.shape[0]):
		avg_log_probs[i] = max_log_probs[i].sum() / masks[i].sum()
	return avg_log_probs

def masked_cross_entropy(logits, target, length):
	length = Variable(torch.LongTensor(length))
	if USE_CUDA:
		length = length.cuda()

	"""
	Args:
		logits: A Variable containing a FloatTensor of size
			(batch, max_len, num_classes) which contains the
			unnormalized probability for each class.
		target: A Variable containing a LongTensor of size
			(batch, max_len) which contains the index of the true
			class for each corresponding step.
		length: A Variable containing a LongTensor of size (batch,)
			which contains the length of each data in a batch.

	Returns:
		loss: An average loss value masked by the length.
	"""

	# logits_flat: (batch * max_len, num_classes)
	logits_flat = logits.view(-1, logits.size(-1))
	# log_probs_flat: (batch * max_len, num_classes)
	log_probs_flat = functional.log_softmax(logits_flat, dim=1)
	# target_flat: (batch * max_len, 1)
	target_flat = target.view(-1, 1)
	# losses_flat: (batch * max_len, 1)
	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
	# losses: (batch, max_len)
	losses = losses_flat.view(*target.size())
	# mask: (batch, max_len)
	mask = sequence_mask(sequence_length=length, max_len=target.size(1))
	losses = losses * mask.float()
	loss = losses.sum() / length.float().sum()
	return loss

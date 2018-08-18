import torch
from torch.autograd import Variable
import pdb
from read_pqa_data import *
import random
import numpy as np
from joint_constants import *

# Return a list of indexes, one for each word in the sentence, plus EOS
def prepare_sequence(seq, to_ix, max_len, cuda=False):
	sequence = [to_ix[w] if w in to_ix else to_ix['<unk>'] for w in seq.split(' ')[:(max_len-1)]]
	sequence.append(EOS_token)
	length = len(sequence)
	sequence += [PAD_token]*(max_len - len(sequence))
	return sequence, length

def iterate_minibatches(post_seqs, post_lens, ques_seqs, ques_lens, ans_seqs, ans_lens, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(len(post_seqs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(post_seqs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield post_seqs[excerpt], post_lens[excerpt], \
				ques_seqs[excerpt], ques_lens[excerpt], \
				ans_seqs[excerpt], ans_lens[excerpt]

def reverse_dict(word2index):
	index2word = {}
	for w, ix in word2index.iteritems():
		index2word[ix] = w
	return index2word

def preprocess_data(triples, word2index):
	post_seqs = []
	post_lens = []
	ques_seqs = []
	ques_lens = []
	ans_seqs = []
	ans_lens = []

	# Choose random triples
	for i in range(len(triples)):
		triple = triples[i]
		post_seq, post_len = prepare_sequence(triple[0], word2index, MAX_POST_LEN)
		post_seqs.append(post_seq)
		post_lens.append(post_len)
		ques_seq, ques_len = prepare_sequence(triple[1], word2index, MAX_QUES_LEN)
		ques_seqs.append(ques_seq)
		ques_lens.append(ques_len)
		ans_seq, ans_len = prepare_sequence(triple[2], word2index, MAX_ANS_LEN)
		ans_seqs.append(ans_seq)
		ans_lens.append(ans_len)

	#post_seqs = Variable(torch.LongTensor(np.array(post_seqs)).cuda())
	#ques_seqs = Variable(torch.LongTensor(np.array(ques_seqs)).cuda())
	#ans_seqs = Variable(torch.LongTensor(np.array(ans_seqs)).cuda())

	return post_seqs,post_lens,ques_seqs,ques_lens,ans_seqs,ans_lens

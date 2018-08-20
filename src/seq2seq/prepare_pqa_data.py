import torch
from torch.autograd import Variable
import pdb
from read_pqa_data import *
import random
import numpy as np
from RL_constants import *

# Return a list of indexes, one for each word in the sentence, plus EOS
def prepare_sequence(seq, word2index, max_len):
	sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in seq.split(' ')[:(max_len-1)]]
	sequence.append(word2index[EOS_token])
	length = len(sequence)
	sequence += [word2index[PAD_token]]*(max_len - len(sequence))
	return sequence, length

def prepare_pq_sequence(post_seq, ques_seq, word2index, post_max_len, ques_max_len):
	sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in post_seq.split(' ')[:(post_max_len-1)]]
	sequence.append(word2index[EOP_token])
	sequence += [word2index[w] if w in word2index else word2index['<unk>'] for w in ques_seq.split(' ')[:(ques_max_len-1)]]
	sequence.append(word2index[EOS_token])
	length = len(sequence)
	sequence += [word2index[PAD_token]]*(post_max_len + ques_max_len - len(sequence))	
	return sequence, length

def iterate_minibatches(input_seqs, input_lens, output_seqs, output_lens, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(len(input_seqs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(input_seqs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield input_seqs[excerpt], input_lens[excerpt], \
				output_seqs[excerpt], output_lens[excerpt] 

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
	post_ques_seqs = []
	post_ques_lens = []
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
		post_ques_seq, post_ques_len = prepare_pq_sequence(triple[0], triple[1], word2index, MAX_POST_LEN, MAX_QUES_LEN)
		post_ques_seqs.append(post_ques_seq)
		post_ques_lens.append(post_ques_len)
		ans_seq, ans_len = prepare_sequence(triple[2], word2index, MAX_ANS_LEN)
		ans_seqs.append(ans_seq)
		ans_lens.append(ans_len)

	return post_seqs, post_lens, ques_seqs, ques_lens, \
			post_ques_seqs, post_ques_lens, ans_seqs, ans_lens

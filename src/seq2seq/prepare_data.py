import torch
from torch.autograd import Variable
import pdb
from read_data import *
import numpy as np
from constants import *


# Return a list of indexes, one for each word in the sentence, plus EOS
def prepare_sequence(seq, word2index, max_len):
	sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in seq.split(' ')[:max_len-1]]
	sequence.append(word2index[EOS_token])
	length = len(sequence)
	sequence += [word2index[PAD_token]]*(max_len - len(sequence))
	return sequence, length


def prepare_pq_sequence(post_seq, ques_seq, word2index, max_post_len, max_ques_len):
	p_sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in post_seq.split(' ')[:(max_post_len-1)]]
	p_sequence.append(word2index[EOP_token])
	p_sequence += [word2index[PAD_token]]*(max_post_len - len(p_sequence))	
	q_sequence = [word2index[w] if w in word2index else word2index['<unk>'] for w in ques_seq.split(' ')[:(max_ques_len-1)]]
	q_sequence.append(word2index[EOS_token])
	q_sequence += [word2index[PAD_token]]*(max_ques_len - len(q_sequence))	
	sequence = p_sequence + q_sequence
	length = max_post_len, max_ques_len
	return sequence, length


def preprocess_data(triples, word2index, max_post_len, max_ques_len, max_ans_len):
	id_seqs = []
	post_seqs = []
	post_lens = []
	ques_seqs = []
	ques_lens = []
	post_ques_seqs = []
	post_ques_lens = []
	ans_seqs = []
	ans_lens = []

	for i in range(len(triples)):
		curr_id, post, ques, ans = triples[i]
		id_seqs.append(curr_id)
		post_seq, post_len = prepare_sequence(post, word2index, max_post_len)
		post_seqs.append(post_seq)
		post_lens.append(post_len)
		ques_seq, ques_len = prepare_sequence(ques, word2index, max_ques_len)
		ques_seqs.append(ques_seq)
		ques_lens.append(ques_len)
		post_ques_seq, post_ques_len = prepare_pq_sequence(post, ques, word2index, max_post_len, max_ques_len)
		post_ques_seqs.append(post_ques_seq)
		post_ques_lens.append(post_ques_len)
		if ans is not None:
			ans_seq, ans_len = prepare_sequence(ans, word2index, max_ans_len)
			ans_seqs.append(ans_seq)
			ans_lens.append(ans_len)

	return id_seqs, post_seqs, post_lens, ques_seqs, ques_lens, \
		post_ques_seqs, post_ques_lens, ans_seqs, ans_lens

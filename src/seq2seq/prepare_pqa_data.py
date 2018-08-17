import torch
from torch.autograd import Variable
from read_pqa_data import *
import random
import numpy as np
from constants import *

def prepare_data(train_context, train_question, train_answer, test_context, test_question, test_answer):
	p_data, q_data, a_data, train_triples = read_data(train_context, train_question, train_answer, split='train')
	test_triples = read_data(test_context, test_question, test_answer, split='test')

	triples = train_triples + test_triples
	print("Indexing words...")
	for triple in triples:
		p_data.index_words(triple[0])
		q_data.index_words(triple[1])
		a_data.index_words(triple[2])
	
	print('Indexed %d words in post input, %d words in ques, %d words in ans' % (p_data.n_words, q_data.n_words, a_data.n_words))

	p_data.trim_using_tfidf()
	q_data.trim(MIN_COUNT)
	a_data.trim(MIN_COUNT)

	return p_data, q_data, a_data, train_triples, test_triples

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
	indices = []
	for word in sentence.split(' '):
	#for word in sentence:
		if word in lang.word2index:
			indices.append(lang.word2index[word])
	indices.append(EOS_token)
	return indices

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
	if len(seq) >= max_length:
		return seq[:max_length]
	seq += [PAD_token for i in range(max_length - len(seq))]
	return seq

def iterate_minibatches(post_seqs, ques_seqs, ans_seqs, batch_size):
	for start_idx in range(0, len(post_seqs) - batch_size + 1, batch_size):
		excerpt = slice(start_idx, start_idx + batch_size)
		yield post_seqs[excerpt], ques_seqs[excerpt], ans_seqs[excerpt]

def add_padding(post_seqs, ques_seqs, ans_seqs, USE_CUDA):
	# For input and target sequences, get array of lens and pad with 0s to max length
	post_lens = [min(len(s), MAX_POST_LEN) for s in post_seqs]
	post_padded = [pad_seq(s, MAX_POST_LEN) for s in post_seqs]

	ques_lens = [min(len(s), MAX_QUES_LEN) for s in ques_seqs]
	ques_padded = [pad_seq(s, MAX_QUES_LEN) for s in ques_seqs]

	ans_lens = [min(len(s), MAX_ANS_LEN) for s in ans_seqs]
	ans_padded = [pad_seq(s, MAX_ANS_LEN) for s in ans_seqs]

	post_var = Variable(torch.LongTensor(np.array(post_padded)).cuda()).transpose(0, 1)
	ques_var = Variable(torch.LongTensor(np.array(ques_padded)).cuda()).transpose(0, 1)
	ans_var = Variable(torch.LongTensor(np.array(ans_padded)).cuda()).transpose(0, 1)

	return post_var, post_lens, ques_var, ques_lens, ans_var, ans_lens

def preprocess_data(p_data, q_data, a_data, triples, shuffle):
	post_seqs = []
	ques_seqs = []
	ans_seqs = []

	if shuffle:
		random.shuffle(triples)
	# Choose random triples
	for i in range(len(triples)):
		triple = triples[i]
		post_seqs.append(indexes_from_sentence(p_data, triple[0]))
		ques_seqs.append(indexes_from_sentence(q_data, triple[1]))
		ans_seqs.append(indexes_from_sentence(a_data, triple[2]))

	if shuffle:
		# Zip into triples, sort by length (descending), unzip
		seq_triples = sorted(zip(post_seqs, ques_seqs, ans_seqs), key=lambda p: len(p[0]), reverse=True)
		post_seqs, ques_seqs, ans_seqs = zip(*seq_triples)
	
	return post_seqs, ques_seqs, ans_seqs

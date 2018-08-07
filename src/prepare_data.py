import torch
from torch.autograd import Variable
from read_data import *
import random
import numpy as np
from constants import *

def prepare_data(post_data_tsv, qa_data_tsv, train_ids_file, test_ids_file):
	p_data, q_data, train_triples, test_triples = read_data(post_data_tsv, qa_data_tsv, train_ids_file, test_ids_file)

	triples = train_triples + test_triples
	print("Indexing words...")
	for triple in triples:
		p_data.index_words(triple[0])
		q_data.index_words(triple[1])
	
	print('Indexed %d words in post input, %d words in ques' % (p_data.n_words, q_data.n_words))

	p_data.trim_using_tfidf()
	q_data.trim(MIN_COUNT)

	return p_data, q_data, train_triples, test_triples

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

def iterate_minibatches(input_seqs, target_seqs, batch_size):
	for start_idx in range(0, len(input_seqs) - batch_size + 1, batch_size):
		excerpt = slice(start_idx, start_idx + batch_size)
		yield input_seqs[excerpt], target_seqs[excerpt]

def add_padding(input_seqs, target_seqs, USE_CUDA):
	# For input and target sequences, get array of lengths and pad with 0s to max length
	input_lengths = [min(len(s), MAX_POST_LEN) for s in input_seqs]
	input_padded = [pad_seq(s, MAX_POST_LEN) for s in input_seqs]

	target_lengths = [min(len(s), MAX_QUES_LEN) for s in target_seqs]
	target_padded = [pad_seq(s, MAX_QUES_LEN) for s in target_seqs]

	if USE_CUDA:
		input_var = Variable(torch.LongTensor(np.array(input_padded)).cuda()).transpose(0, 1)
		target_var = Variable(torch.LongTensor(np.array(target_padded)).cuda()).transpose(0, 1)
	else:
		input_var = Variable(torch.LongTensor(np.array(input_padded))).transpose(0, 1)
        target_var = Variable(torch.LongTensor(np.array(target_padded))).transpose(0, 1)
	
	target_var = Variable(torch.LongTensor(np.array(target_padded)).cuda()).transpose(0, 1)
	return input_var, input_lengths, target_var, target_lengths

def preprocess_data(p_data, q_data, triples):
	input_seqs = []
	target_seqs = []

	# Choose random triples
	for i in range(len(triples)):
		triple = random.choice(triples)
		input_seqs.append(indexes_from_sentence(p_data, triple[0]))
		target_seqs.append(indexes_from_sentence(q_data, triple[1]))

	# Zip into triples, sort by length (descending), unzip
	seq_triples = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
	input_seqs, target_seqs = zip(*seq_triples)
	
	return input_seqs, target_seqs

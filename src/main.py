import argparse
import string
import random
import time
import datetime
import math
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *

import numpy as np

from prepare_data import *
from encoderRNN import *
from encoderAvgEmb import *
from attnDecoderRNN import *
from helper import *
from attn import *
from read_data import *
from train import *
from evaluate import *

from constants import *

def load_pretrained_emb(word_vec_fname, p_data):
	pretrained_emb = np.random.rand(p_data.n_words, word_emb_size)
	word_vec_file = open(word_vec_fname, 'r')
	for line in word_vec_file.readlines():
		vals = line.rstrip().split(' ')
		word = vals[0]
		if word in p_data.word2index:
			pretrained_emb[p_data.word2index[word]] = map(float, vals[1:])
	return pretrained_emb

def main(args):

	p_data, q_data, train_triples, test_triples = prepare_data(args.post_data_tsvfile, args.qa_data_tsvfile, \
																args.train_ids_file, args.test_ids_file, args.sim_ques_fname)

	pretrained_emb = load_pretrained_emb(args.word_vec_fname, p_data)

	#N = int(len(triples)*0.8)
	#train_triples = triples[:N]
	#test_triples = triples[N:]

	# Initialize models
	#p_encoder = EncoderAvgEmb(pretrained_emb)
	p_encoder = EncoderRNN(p_data.n_words, hidden_size, n_layers, dropout=dropout)
	q_encoder = EncoderRNN(q_data.n_words, hidden_size, n_layers, dropout=dropout)
	decoder = AttnDecoderRNN(attn_model, hidden_size, q_data.n_words, n_layers)

	# Initialize optimizers and criterion
	p_encoder_optimizer = optim.Adam(p_encoder.parameters(), lr=learning_rate)
	q_encoder_optimizer = optim.Adam(q_encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	criterion = nn.CrossEntropyLoss()

	# Move models to GPU
	if USE_CUDA:
		p_encoder.cuda()
		q_encoder.cuda()
		decoder.cuda()

	# Keep track of time elapsed and running averages
	start = time.time()
	print_loss_total = 0 # Reset every print_every
	epoch = 0.0
	
	print 'No. of train_triples %d' % len(train_triples)
	print 'No. of test_triples %d' % len(test_triples)

	#p_input_batches, p_input_lengths, q_input_batches, q_input_lengths, target_batches, target_lengths = \
	#		preprocess_data(p_data, q_data, train_triples)

	p_input_seqs, q_input_seqs, target_seqs = preprocess_data(p_data, q_data, train_triples)

	n_batches = len(train_triples) / batch_size
	while epoch < n_epochs:
		epoch += 1
		
		#for p_batch, p_lens, q_batch, q_lens, target_batch, target_lens in iterate_minibatches(p_input_batches, p_input_lengths, \
		#				q_input_batches, q_input_lengths, target_batches, target_lengths, batch_size):
	
		for p_input_seqs_batch, q_input_seqs_batch, target_seqs_batch in \
				iterate_minibatches(p_input_seqs, q_input_seqs, target_seqs, batch_size):	

			p_batch, p_lens, q_batch, q_lens, target_batch, target_lens = \
				add_padding(p_input_seqs_batch, q_input_seqs_batch, target_seqs_batch, USE_CUDA)

			# Run the train function
			loss, ec, dc = train(
				p_batch, p_lens, 
				q_batch, q_lens, 
				target_batch, target_lens,
				p_encoder, q_encoder, decoder,
				p_encoder_optimizer, q_encoder_optimizer, decoder_optimizer, criterion, USE_CUDA
			)
	
			# Keep track of loss
			print_loss_total += loss
		
		print_loss_avg = print_loss_total / n_batches
		print_loss_total = 0
		print_summary = '%s %d %.4f' % (time_since(start, epoch / n_epochs), epoch, print_loss_avg)
		print(print_summary)
		evaluate_randomly(p_data, q_data, test_triples, p_encoder, q_encoder, decoder)
	
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_data_tsvfile", type = str)
	argparser.add_argument("--qa_data_tsvfile", type = str)
	argparser.add_argument("--train_ids_file", type = str)
	argparser.add_argument("--test_ids_file", type = str)
	argparser.add_argument("--sim_ques_fname", type = str)
	argparser.add_argument("--word_vec_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	

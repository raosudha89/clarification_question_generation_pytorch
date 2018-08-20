import argparse
import os
import cPickle as p
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

import numpy as np

from seq2seq.read_pqa_data import *
from seq2seq.prepare_pqa_data import *
from seq2seq.encoderRNN import *
from seq2seq.attnDecoderRNN import *
from seq2seq.masked_cross_entropy import *
from seq2seq.helper import *
from seq2seq.attn import *
from seq2seq.RL_train import *
from seq2seq.RL_evaluate import *

from RL_constants import *

def update_embs(word2index, word_embeddings):
	word_embeddings[word2index[PAD_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
	word_embeddings[word2index[SOS_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
	word_embeddings[word2index[EOP_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
	word_embeddings[word2index[EOS_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
	return word_embeddings

def main(args):
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.array(word_embeddings)
	word2index = p.load(open(args.vocab, 'rb'))

	word_embeddings = update_embs(word2index, word_embeddings)

	index2word = reverse_dict(word2index)

	train_data = read_data(args.train_context, args.train_question, args.train_answer, split='train')
	test_data = read_data(args.test_context, args.test_question, args.test_answer, split='test')

	print 'No. of train_data %d' % len(train_data)
	print 'No. of test_data %d' % len(test_data)

	post_seqs, post_lens, ques_seqs, ques_lens, \
            post_ques_seqs, post_ques_lens, ans_seqs, ans_lens = preprocess_data(train_data, word2index)

	q_train_data = post_seqs, post_lens, ques_seqs, ques_lens
	a_train_data = post_ques_seqs, post_ques_lens, ans_seqs, ans_lens

	post_seqs, post_lens, ques_seqs, ques_lens, \
            post_ques_seqs, post_ques_lens, ans_seqs, ans_lens = preprocess_data(test_data, word2index)

	q_test_data = post_seqs, post_lens, ques_seqs, ques_lens
	a_test_data = post_ques_seqs, post_ques_lens, ans_seqs, ans_lens

	run_model(q_train_data, q_test_data, word2index, index2word, \
			word_embeddings, args.test_pred_question, args.q_encoder_params, args.q_decoder_params)

	run_model(a_train_data, a_test_data, word2index, index2word, \
			word_embeddings, args.test_pred_question, args.a_encoder_params, args.a_decoder_params)

def run_model(train_data, test_data, word2index, index2word, \
			word_embeddings, out_file, encoder_params_file, decoder_params_file):
	# Initialize q models
	encoder = EncoderRNN(len(word2index), hidden_size, word_embeddings, n_layers, dropout=dropout)
	decoder = AttnDecoderRNN(attn_model, hidden_size, len(word2index), word_embeddings, n_layers)

	#if os.path.isfile(encoder_params_file):
	#	print 'Loading saved params...'
	#	encoder.load_state_dict(torch.load(encoder_params_file))
	#	decoder.load_state_dict(torch.load(decoder_params_file))
	#	print 'Done!'

	# Initialize optimizers
	encoder_optimizer = optim.Adam([par for par in encoder.parameters() if par.requires_grad], lr=learning_rate)
	decoder_optimizer = optim.Adam([par for par in decoder.parameters() if par.requires_grad], lr=learning_rate * decoder_learning_ratio)

	# Move models to GPU
	if USE_CUDA:
		encoder.cuda()
		decoder.cuda()

	# Keep track of time elapsed and running averages
	start = time.time()
	print_loss_total = 0 # Reset every print_every
	epoch = 0.0

	input_seqs, input_lens, output_seqs, output_lens = train_data
	
	n_batches = len(input_seqs) / batch_size
	while epoch < n_epochs:
		epoch += 1
		for input_seqs_batch, input_lens_batch, \
			output_seqs_batch, output_lens_batch in \
                iterate_minibatches(input_seqs, input_lens, output_seqs, output_lens, batch_size):

			start_time = time.time()
			# Run the train function
			loss = train(
				input_seqs_batch, input_lens_batch,
				output_seqs_batch, output_lens_batch,
				encoder, decoder,
				encoder_optimizer, decoder_optimizer, word2index[SOS_token]
			)
	
			# Keep track of loss
			print_loss_total += loss
		
		print_loss_avg = print_loss_total / n_batches
		print_loss_total = 0
		print_summary = '%s %d %.4f' % (time_since(start, epoch / n_epochs), epoch, print_loss_avg)
		print(print_summary)
		print 'Epoch: %d' % epoch
		if epoch == n_epochs-1:
			print 'Saving model params'
			torch.save(encoder.state_dict(), encoder_params_file)
			torch.save(decoder.state_dict(), decoder_params_file)
		if epoch > n_epochs - 10:
			out_file = open(args.test_pred_question+'.epoch%d' % int(epoch), 'w')	
			evaluate(word2index, index2word, encoder, decoder, test_data, batch_size, out_file)
	
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--train_context", type = str)
	argparser.add_argument("--train_question", type = str)
	argparser.add_argument("--train_answer", type = str)
	argparser.add_argument("--tune_context", type = str)
	argparser.add_argument("--tune_question", type = str)
	argparser.add_argument("--tune_answer", type = str)
	argparser.add_argument("--test_context", type = str)
	argparser.add_argument("--test_question", type = str)
	argparser.add_argument("--test_answer", type = str)
	argparser.add_argument("--test_pred_question", type = str)
	argparser.add_argument("--test_pred_answer", type = str)
	argparser.add_argument("--q_encoder_params", type = str)
	argparser.add_argument("--q_decoder_params", type = str)
	argparser.add_argument("--a_encoder_params", type = str)
	argparser.add_argument("--a_decoder_params", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--cuda", type = bool)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	

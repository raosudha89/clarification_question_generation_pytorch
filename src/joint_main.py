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

import numpy as np

from seq2seq.read_pqa_data import *
from seq2seq.prepare_pqa_data import *
from seq2seq.encoderRNN import *
from seq2seq.attnDecoderRNN import *
from seq2seq.masked_cross_entropy import *
from seq2seq.helper import *
from seq2seq.attn import *
from seq2seq.joint_train import *
from seq2seq.joint_evaluate import *

from constants import *

def main(args):

	p_data, q_data, a_data, train_data, test_data = prepare_data(args.train_context, args.train_question, args.train_answer, \
																	args.test_context, args.test_question, args.test_answer)

	# Initialize q models
	q_encoder = EncoderRNN(p_data.n_words, hidden_size, n_layers, dropout=dropout)
	q_decoder = AttnDecoderRNN(attn_model, hidden_size, q_data.n_words, n_layers)

	# Initialize a models
	a_encoder = EncoderRNN(q_data.n_words, hidden_size, n_layers, dropout=dropout)
	a_decoder = AttnDecoderRNN(attn_model, hidden_size, a_data.n_words, n_layers)

	# Initialize optimizers and criterion
	q_encoder_optimizer = optim.Adam(q_encoder.parameters(), lr=learning_rate)
	q_decoder_optimizer = optim.Adam(q_decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	q_criterion = nn.CrossEntropyLoss()

	# Initialize optimizers and criterion
	a_encoder_optimizer = optim.Adam(a_encoder.parameters(), lr=learning_rate)
	a_decoder_optimizer = optim.Adam(a_decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	a_criterion = nn.CrossEntropyLoss()

	# Move models to GPU
	if USE_CUDA:
		q_encoder.cuda()
		q_decoder.cuda()
		a_encoder.cuda()
		a_decoder.cuda()

	# Keep track of time elapsed and running averages
	start = time.time()
	print_loss_total = 0 # Reset every print_every
	epoch = 0.0
	
	print 'No. of train_data %d' % len(train_data)
	print 'No. of test_data %d' % len(test_data)

	post_seqs, ques_seqs, ans_seqs = preprocess_data(p_data, q_data, a_data, train_data, shuffle=True)

	n_batches = len(train_data) / batch_size
	while epoch < n_epochs:
		epoch += 1
		if epoch > n_epochs - 10:
			ques_out_file = open(args.test_pred_question+'.epoch%d' % int(epoch), 'w')	
			ans_out_file = open(args.test_pred_answer+'.epoch%d' % int(epoch), 'w')	
		else:
			ques_out_file = None
			ans_out_file = None
		for post_seqs_batch, ques_seqs_batch, ans_seqs_batch in \
				iterate_minibatches(post_seqs, ques_seqs, ans_seqs, batch_size):	

			post_seqs_batch, post_lens, \
			ques_seqs_batch, ques_lens, \
			ans_seqs_batch, ans_lens = add_padding(post_seqs_batch, ques_seqs_batch, ans_seqs_batch, USE_CUDA)

			start_time = time.time()
			# Run the train function
			loss = train(
				post_seqs_batch, post_lens,
				ques_seqs_batch, ques_lens,
				ans_seqs_batch, ans_lens,
				q_encoder, q_decoder,
				q_encoder_optimizer, q_decoder_optimizer, q_criterion,
				a_encoder, a_decoder,
				a_encoder_optimizer, a_decoder_optimizer, a_criterion, USE_CUDA
			)
	
			# Keep track of loss
			print_loss_total += loss
		
		print_loss_avg = print_loss_total / n_batches
		print_loss_total = 0
		print_summary = '%s %d %.4f' % (time_since(start, epoch / n_epochs), epoch, print_loss_avg)
		print(print_summary)
		print 'Epoch: %d' % epoch
		evaluate_testset(p_data, q_data, a_data, q_encoder, q_decoder, \
						 a_encoder, a_decoder, test_data, batch_size, ques_out_file, ans_out_file)
	
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
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	

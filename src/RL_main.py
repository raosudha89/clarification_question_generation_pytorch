import argparse
import os
import cPickle as p
import string
import time
import datetime
import math
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

import numpy as np

from seq2seq.encoderRNN import *
from seq2seq.attnDecoderRNN import *
from seq2seq.read_data import *
from seq2seq.prepare_data import *
from seq2seq.masked_cross_entropy import *
from seq2seq.RL_train import *
from utility.RNN import *
from utility.FeedForward import *
from utility.RL_evaluate import *
from RL_helper import *
from constants import *

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
	#word_embeddings = update_embs(word2index, word_embeddings) --> updating embs gives poor utility results (0.5 acc)
	index2word = reverse_dict(word2index)

	train_data = read_data(args.train_context, args.train_question, args.train_answer, \
							args.max_post_len, args.max_ques_len, args.max_ans_len, split='train')
	test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, \
							args.max_post_len, args.max_ques_len, args.max_ans_len, split='test')
	#test_data = read_data(args.test_context, args.test_question, args.test_answer, \
	#						args.max_post_len, args.max_ques_len, args.max_ans_len, split='test')

	print 'No. of train_data %d' % len(train_data)
	print 'No. of test_data %d' % len(test_data)
	run_model(train_data, test_data, word_embeddings, word2index, args)

def run_model(train_data, test_data, word_embeddings, word2index, args):
	tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens = preprocess_data(train_data, word2index, \
																				args.max_post_len, args.max_ques_len, args.max_ans_len)

	te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, \
    te_post_ques_seqs, te_post_ques_lens, te_ans_seqs, te_ans_lens = preprocess_data(test_data, word2index, \
																				args.max_post_len, args.max_ques_len, args.max_ans_len)

	q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT).cuda()
	q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2).cuda()

	a_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT).cuda()
	a_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2).cuda()

	# Load encoder, decoder params
	#q_encoder.load_state_dict(torch.load(args.q_encoder_params))
	#q_decoder.load_state_dict(torch.load(args.q_decoder_params))
	#a_encoder.load_state_dict(torch.load(args.a_encoder_params))
	#a_decoder.load_state_dict(torch.load(args.a_decoder_params))

	q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	a_encoder_optimizer = optim.Adam([par for par in a_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	a_decoder_optimizer = optim.Adam([par for par in a_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1).cuda()
	question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1).cuda()
	answer_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1).cuda()
	utility_model = FeedForward(HIDDEN_SIZE*3).cuda()

	# Load utility calculator model params
	context_model.load_state_dict(torch.load(args.context_params))
	question_model.load_state_dict(torch.load(args.question_params))
	answer_model.load_state_dict(torch.load(args.answer_params))
	utility_model.load_state_dict(torch.load(args.utility_params))

	epoch = 0.
	start = time.time()
	n_batches = len(tr_post_seqs)/args.batch_size
	while epoch < args.n_epochs:
		epoch += 1
		total_loss = 0.
		for post, pl, ques, ql, pq, pql, a, al in iterate_minibatches(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
													tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens, args.batch_size):

			q_loss, q_log_probs, q_pred, ql_pred = train(post, pl, ques, ql, q_encoder, q_decoder, \
															q_encoder_optimizer, q_decoder_optimizer, \
															word2index, args.max_ques_len, args.batch_size)
			pq_pred = np.concatenate((post, q_pred), axis=1)
			pql_pred = np.full((args.batch_size), args.max_post_len+args.max_ques_len)
			a_loss, a_log_probs, a_pred, al_pred = train(pq_pred, pql_pred, a, al, a_encoder, a_decoder, \
															a_encoder_optimizer, a_decoder_optimizer, \
															word2index, args.max_ans_len, args.batch_size)

			u_preds = evaluate(context_model, question_model, answer_model, utility_model, post, q_pred, a_pred)	
			log_probs = torch.add(q_log_probs, a_log_probs)
			loss = 0.
			for b in range(args.batch_size):
				loss += log_probs[b]*u_preds[b].item()
			loss = -1.0*loss
			total_loss += loss
			loss.backward()
			q_encoder_optimizer.step()
			q_decoder_optimizer.step()
			a_encoder_optimizer.step()
			a_decoder_optimizer.step()
		print_loss_avg = total_loss / n_batches
		print_summary = '%s %d %.4f' % (time_since(start, epoch / args.n_epochs), epoch, print_loss_avg)
		print(print_summary)
		

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
	argparser.add_argument("--context_params", type = str)
	argparser.add_argument("--question_params", type = str)
	argparser.add_argument("--answer_params", type = str)
	argparser.add_argument("--utility_params", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--cuda", type = bool, default=True)
	argparser.add_argument("--max_post_len", type = int, default=300)
	argparser.add_argument("--max_ques_len", type = int, default=50)
	argparser.add_argument("--max_ans_len", type = int, default=50)
	argparser.add_argument("--n_epochs", type = int, default=20)
	argparser.add_argument("--batch_size", type = int, default=128)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	

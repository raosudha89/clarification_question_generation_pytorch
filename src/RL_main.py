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

	train_data = read_data(args.train_context, args.train_question, args.train_answer, split='train')
	test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, split='test')
	#test_data = read_data(args.test_context, args.test_question, args.test_answer, split='test')

	print 'No. of train_data %d' % len(train_data)
	print 'No. of test_data %d' % len(test_data)

	run_model(train_data, test_data, word2index, word_embeddings)

def run_model(train_data, test_data, word2index, word_embeddings):
	tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens = preprocess_data(train_data, word2index)

	te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, \
    te_post_ques_seqs, te_post_ques_lens, te_ans_seqs, te_ans_lens = preprocess_data(test_data, word2index)

	q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, N_LAYERS, dropout=dropout)
	q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, N_LAYERS)

	a_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, N_LAYERS, dropout=dropout)
	a_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, N_LAYERS)

	# Load encoder, decoder params
	q_encoder.load_state_dict(torch.load(args.q_encoder_params))
	q_decoder.load_state_dict(torch.load(args.q_decoder_params))
	a_encoder.load_state_dict(torch.load(args.a_encoder_params))
	a_decoder.load_state_dict(torch.load(args.a_decoder_params))

	q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	a_encoder_optimizer = optim.Adam([par for par in a_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	a_decoder_optimizer = optim.Adam([par for par in a_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	if USE_CUDA:
		q_encoder.cuda()
		q_decoder.cuda()
		a_encoder.cuda()
		a_decoder.cuda()

	context_model = RNN(len(word_embeddings), len(word_embeddings[0]))
	question_model = RNN(len(word_embeddings), len(word_embeddings[0]))
	answer_model = RNN(len(word_embeddings), len(word_embeddings[0]))
	utility_model = FeedForward(HIDDEN_SIZE*3)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	context_model = context_model.to(device)
	question_model = question_model.to(device)
	answer_model = answer_model.to(device)
	utility_model = utility_model.to(device)

	# Load utility calculator model params
	context_model.load_state_dict(torch.load(args.context_params))
	question_model.load_state_dict(torch.load(args.question_params))
	answer_model.load_state_dict(torch.load(args.answer_params))
	utility_model.load_state_dict(torch.load(args.utility_params))

	epoch = 0
	n_batches = len(tr_post_seqs)/BATCH_SIZE
	while epoch < n_epochs:
		epoch += 1
		for p, pl, q, ql, pq, pql, a, al in iterate_minibatches(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
													tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens, BATCH_SIZE):

			q_loss, q_log_probs, q_pred, ql_pred = train(p, pl, q, ql, q_encoder, q_decoder, \
													q_encoder_optimizer, q_decoder_optimizer, word2index, MAX_QUES_LEN)	
			pq_pred = np.concatenate((p, q_pred), axis=1)
			pql_pred = np.full((BATCH_SIZE), MAX_POST_LEN+MAX_QUES_LEN)
			#a_loss, a_pred_seqs, a_pred_lens = train(pq, pql, a, al, a_encoder, a_decoder, \
			#										a_encoder_optimizer, a_decoder_optimizer, word2index, MAX_ANS_LEN)
			a_loss, a_log_probs, a_pred, al_pred = train(pq_pred, pql_pred, a, al, a_encoder, a_decoder, \
													a_encoder_optimizer, a_decoder_optimizer, word2index, MAX_ANS_LEN)

			u_preds = evaluate(context_model, question_model, answer_model, utility_model, p, q_pred, a_pred)	
			log_probs = torch.add(q_log_probs, a_log_probs)
			loss = 0.
			for b in range(BATCH_SIZE):
				loss += log_probs[b]*u_preds[b].item()
			loss = -1.0*loss
			#loss = q_loss + a_loss
			print loss
			loss.backward()
			q_encoder_optimizer.step()
			q_decoder_optimizer.step()
			a_encoder_optimizer.step()
			a_decoder_optimizer.step()

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
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	

import argparse
import sys
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
from seq2seq.RL_evaluate import *
from seq2seq.RL_inference import *
from utility.RNN import *
from utility.FeedForward import *
from utility.RL_evaluate import *
from utility.RL_train import *
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
	run_model(train_data, test_data, word_embeddings, word2index, index2word, args)

def run_model(train_data, test_data, word_embeddings, word2index, index2word, args):
	print 'Preprocessing train data..'
	tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens = preprocess_data(train_data, word2index, \
																				args.max_post_len, args.max_ques_len, args.max_ans_len)

	print 'Preprocessing test data..'
	te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, \
    te_post_ques_seqs, te_post_ques_lens, te_ans_seqs, te_ans_lens = preprocess_data(test_data, word2index, \
																				args.max_post_len, args.max_ques_len, args.max_ans_len)

	print 'Defining encoder decoder models'
	q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
	q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

	a_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
	a_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

	if USE_CUDA:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		q_encoder = q_encoder.to(device)
		q_decoder = q_decoder.to(device)
		a_encoder = a_encoder.to(device)
		a_decoder = a_decoder.to(device)

	# Load encoder, decoder params
	print 'Loading encoded, decoder params'
	q_encoder.load_state_dict(torch.load(args.q_encoder_params))
	q_decoder.load_state_dict(torch.load(args.q_decoder_params))
	a_encoder.load_state_dict(torch.load(args.a_encoder_params))
	a_decoder.load_state_dict(torch.load(args.a_decoder_params))

	q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	a_encoder_optimizer = optim.Adam([par for par in a_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	a_decoder_optimizer = optim.Adam([par for par in a_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
	question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
	answer_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
	utility_model = FeedForward(HIDDEN_SIZE*3)

	#u_optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) + \
	#						list([par for par in question_model.parameters() if par.requires_grad]) + \
	#						list([par for par in answer_model.parameters() if par.requires_grad]) + \
	#						list([par for par in utility_model.parameters() if par.requires_grad]))

	if USE_CUDA:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		context_model.to(device)
		question_model.to(device)
		answer_model.to(device)
		utility_model.to(device)	

	# Load utility calculator model params
	print 'Loading utility model params'
	context_model.load_state_dict(torch.load(args.context_params))
	question_model.load_state_dict(torch.load(args.question_params))
	answer_model.load_state_dict(torch.load(args.answer_params))
	utility_model.load_state_dict(torch.load(args.utility_params))

	for param in context_model.parameters(): param.requires_grad = False
	for param in question_model.parameters(): param.requires_grad = False
	for param in answer_model.parameters(): param.requires_grad = False
	for param in utility_model.parameters(): param.requires_grad = False

	epoch = 0.
	start = time.time()
	out_file = open(args.test_pred_question+'.pretrained', 'w')	
	#out_file = None
	evaluate_seq2seq(word2index, index2word, q_encoder, q_decoder, \
					te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, args.batch_size, out_file)
					#tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, args.batch_size, out_file)

	n_batches = len(tr_post_seqs)/args.batch_size
	while epoch < args.n_epochs:
		epoch += 1
		total_loss = 0.
		total_u_pred = 0.
		total_u_true_pred = 0.
		for post, pl, ques, ql, pq, pql, ans, al in iterate_minibatches(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
													tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens, args.batch_size):
			q_log_probs, q_pred, ql_pred = train(post, pl, ques, ql, q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer, \
														word2index, args.max_ques_len, args.batch_size)
			g_q_log_probs, g_q_pred, g_ql_pred = inference(post, pl, ques, ql, q_encoder, q_decoder, \
															word2index, args.max_ques_len, args.batch_size)

			pq_pred = np.concatenate((post, q_pred), axis=1)
			pql_pred = np.full((args.batch_size), args.max_post_len+args.max_ques_len)
			a_log_probs, a_pred, al_pred = train(pq_pred, pql_pred, ans, al, a_encoder, a_decoder, \
															a_encoder_optimizer, a_decoder_optimizer, \
															word2index, args.max_ans_len, args.batch_size)
			g_pq_pred = np.concatenate((post, g_q_pred), axis=1)
			g_pql_pred = np.full((args.batch_size), args.max_post_len+args.max_ques_len)
			g_a_log_probs, g_a_pred, g_al_pred = inference(g_pq_pred, g_pql_pred, ans, al, a_encoder, a_decoder, \
															word2index, args.max_ans_len, args.batch_size)
			log_probs = torch.add(q_log_probs, a_log_probs)

			#u_loss, u_true_preds = train_utility(context_model, question_model, answer_model, utility_model, u_optimizer, post, ques, ans)
			u_preds = evaluate_utility(context_model, question_model, answer_model, utility_model, post, q_pred, a_pred)	
			g_u_preds = evaluate_utility(context_model, question_model, answer_model, utility_model, post, g_q_pred, g_a_pred)	
			total_u_pred += u_preds.data.sum() / args.batch_size
			total_u_true_pred += g_u_preds.data.sum() / args.batch_size
			reward = u_preds
			g_reward = g_u_preds

			#reward = calculate_bleu(ques, ql, q_pred, ql_pred, index2word)
			#g_reward = calculate_bleu(ques, ql, g_q_pred, g_ql_pred, index2word)
			#total_u_pred += reward.sum() / args.batch_size 
			#total_u_true_pred += g_reward.sum() / args.batch_size 

			loss = torch.sum(-1.0*log_probs * (reward.data-g_reward.data))
			#loss += u_loss
			#loss += q_loss
			#print g_q_log_probs
			#print g_reward.data
			#loss = -1.0*torch.sum(g_q_log_probs * g_reward.data)

			total_loss += loss
			loss.backward()
			q_encoder_optimizer.step()
			q_decoder_optimizer.step()
			a_encoder_optimizer.step()
			a_decoder_optimizer.step()
			#u_optimizer.step()
		print_loss_avg = total_loss / n_batches
		print_u_pred_avg = total_u_pred / n_batches
		print_u_true_pred_avg = total_u_true_pred / n_batches
		print_summary = '%s %d RL_loss: %.4f U_pred: %.4f U_true_pred: %.4f' % \
						(time_since(start, epoch / args.n_epochs), epoch, print_loss_avg, print_u_pred_avg, print_u_true_pred_avg)
		print(print_summary)
		if epoch > args.n_epochs - 10:
			print 'Running evaluation...'
			out_file = open(args.test_pred_question+'.epoch%d' % int(epoch), 'w')	
			#out_file = None
			evaluate_seq2seq(word2index, index2word, q_encoder, q_decoder, \
							te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, args.batch_size, out_file)
							#tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, args.batch_size, out_file)
		

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
	argparser.add_argument("--max_post_len", type = int, default=300)
	argparser.add_argument("--max_ques_len", type = int, default=50)
	argparser.add_argument("--max_ans_len", type = int, default=50)
	argparser.add_argument("--n_epochs", type = int, default=20)
	argparser.add_argument("--batch_size", type = int, default=128)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
	

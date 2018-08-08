# -*- coding: utf-8 -*-
import argparse
import cPickle as p
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
import pdb
import itertools
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

class LSTM_FFnet(nn.Module):
	def __init__(self, embeddings, hidden_dim, input_dim, label_size, batch_size):
		super(LSTM_FFnet, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
		self.context_lstm = nn.LSTM(len(embeddings[0]), hidden_dim)
		self.question_lstm = nn.LSTM(len(embeddings[0]), hidden_dim)
		self.answer_lstm = nn.LSTM(len(embeddings[0]), hidden_dim)
		self.feedforward = nn.Linear(input_dim, hidden_dim)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.context_hidden = self.init_hidden()
		self.question_hidden = self.init_hidden()
		self.answer_hidden = self.init_hidden()
		self.sigmoid = nn.Sigmoid()

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
				autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

	def forward(self, context, question, answer):
		context_embeds = self.word_embeddings(context)
		context_x = context_embeds.view(len(context[0]), self.batch_size, -1)
		question_embeds = self.word_embeddings(question)
		question_x = question_embeds.view(len(question[0]), self.batch_size, -1)
		answer_embeds = self.word_embeddings(answer)
		answer_x = answer_embeds.view(len(answer[0]), self.batch_size, -1)
		context_lstm_out, self.context_hidden = self.lstm(context_x, self.hidden)
		question_lstm_out, self.question_hidden = self.lstm(question_x, self.question_hidden)
		answer_lstm_out, self.answer_hidden = self.lstm(answer_x, self. answer_hidden)
		context_lstm_out = torch.mean(context_lstm_out, 0)
		question_lstm_out = torch.mean(question_lstm_out, 0)
		answer_lstm_out = torch.mean(answer_lstm_out, 0)	
		out = self.feedforward(torch.cat((context_lstm_out, question_lstm_out, answer_lstm_out), 1))
		y = self.hidden2label(out)
		probs = self.sigmoid(y)
		return probs	

def get_accuracy(truth, pred):
	 assert len(truth)==len(pred)
	 right = 0
	 for i in range(len(truth)):
		 if (truth[i] and pred[i] >= 0.5) or (truth[i] == 0 and pred[i] < 0.5):
			 right += 1.0
	 return right/len(truth)

# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, max_len, cuda=False):
	sequences = []
	for s in seq:
		sequence = [to_ix[w] for w in s.split(' ')[:max_len]]
		sequence += [0]*(max_len - len(s.split(' ')))
		sequences.append(sequence)
	if cuda:
		var = autograd.Variable(torch.LongTensor(sequences).cuda())
	else:
		var = autograd.Variable(torch.LongTensor(sequences))
	return var

def prepare_label(labels, cuda=False):
	if cuda:
		var = autograd.Variable(torch.FloatTensor([float(l) for l in labels]).cuda())
	else:
		var = autograd.Variable(torch.FloatTensor([float(l) for l in labels]))
	return var

def iterate_minibatches(contexts, questions, answers, labels, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(contexts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, contexts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield contexts[excerpt], questions[excerpt], answers[excerpt], labels[excerpt]

def count_grad_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
	return sum(p.numel() for p in model.parameters())

def main(args):
	train_data = p.load(open(args.train_data, 'rb'))
	dev_data = p.load(open(args.tune_data, 'rb'))
	test_data = p.load(open(args.test_data, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	vocab = p.load(open(args.vocab, 'rb'))
	EMBEDDING_DIM = len(word_embeddings[0])
	HIDDEN_DIM = 100
	EPOCH = 100
	BATCH_SIZE = 128
	best_dev_acc = 0.0
	model = LSTM_FFnet(embeddings=word_embeddings, hidden_dim=HIDDEN_DIM, input_dim=EMBEDDING_DIM*3, label_size=1, batch_size=BATCH_SIZE)

	if args.cuda:
		model.cuda()

	loss_function = nn.BCELoss()
	print count_grad_parameters(model)
	print count_all_parameters(model)
	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad] ,lr = 1e-3)
	no_up = 0
	for i in range(EPOCH):
		random.shuffle(train_data)
		print('epoch: %d start!' % i)
		train_epoch(model, train_data, loss_function, optimizer, word_embeddings, i, BATCH_SIZE, args.cuda)
		print('now best dev acc:',best_dev_acc)
		dev_acc = evaluate(model, dev_data, loss_function, word_embeddings, i, BATCH_SIZE, 'dev', args.cuda)
		test_acc = evaluate(model, test_data, loss_function, word_embeddings, i, BATCH_SIZE, 'test', args.cuda)
		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			#os.system('rm mr_best_model_acc_*.model')
			#print('New Best Dev!!!')
			#torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(test_acc*10000)) + '.model')
			no_up = 0
		else:
			no_up += 1
			if no_up >= 10:
				exit()

def evaluate(model, data, loss_function, word_embeddings, i, batch_size, cuda, name ='dev'):
	#model.eval()
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = 0

	data = np.array(data)
	data = np.transpose(data)
	contexts_data, questions_data, answers_data, labels_data = data
	for contexts, questions, answers, labels in \
							iterate_minibatches(contexts_data, questions_data, answers_data, labels_data, batch_size, shuffle=True):
		labels = prepare_label(labels, cuda)
		# init model
		model.hidden = model.init_hidden()
		# prepare input
		contexts = prepare_sequence(contexts, word_embeddings, 250, cuda)
		questions = prepare_sequence(questions, word_embeddings, 50, cuda)
		answers = prepare_sequence(answers, word_embeddings, 50, cuda)
		# run LSTM models
		pred = model(contexts)
		pred = pred[:,0]
		# get label from pred
		loss = loss_function(pred, labels)
		acc = get_accuracy(labels.data, pred.data)
		avg_loss += loss.data[0]
		avg_acc += acc
		num_batches += 1
		if num_batches % 100 == 0:
			print('epoch: %d iterations: %d loss :%g' % (i, num_batches, loss.data[0]))
	avg_loss /= num_batches
	avg_acc /= num_batches
	print('epoch: %d done! \n train avg_loss:%g , avg_acc:%g'%(i, avg_loss, avg_acc))
	return avg_acc

def train_epoch(model, train_data, loss_function, optimizer, word_embeddings, i, batch_size, cuda):
	#model.train()
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = 0
	batch_sent = []

	#for context, question, answer, label in train_data:
	train_data = np.array(train_data)
	train_data = np.transpose(train_data)
	contexts_data, questions_data, answers_data, labels_data = train_data
	for contexts, questions, answers, labels in \
							iterate_minibatches(contexts_data, questions_data, answers_data, labels_data, batch_size, shuffle=True):
		labels = prepare_label(labels, cuda)
		# init model
		model.hidden = model.init_hidden()
		# prepare input
		contexts = prepare_sequence(contexts, word_embeddings, 250, cuda)
		questions = prepare_sequence(questions, word_embeddings, 50, cuda)
		answers = prepare_sequence(answers, word_embeddings, 50, cuda)
		# run LSTM models
		pred = model(contexts)
		pred = pred[:,0]
		# get label from pred
		model.zero_grad()
		loss = loss_function(pred, labels)
		acc = get_accuracy(labels.data, pred.data)
		avg_loss += loss.data[0]
		avg_acc += acc
		num_batches += 1
		if num_batches % 100 == 0:
			print('epoch: %d iterations: %d loss :%g' % (i, num_batches, loss.data[0]))
		loss.backward()
		optimizer.step()

	avg_loss /= num_batches
	avg_acc /= num_batches
	print('epoch: %d done! \n train avg_loss:%g , avg_acc:%g'%(i, avg_loss, avg_acc))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--train_data", type = str)
	argparser.add_argument("--tune_data", type = str)
	argparser.add_argument("--test_data", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--cuda", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


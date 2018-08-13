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

class LSTM(nn.Module):
	def __init__(self, embeddings, hidden_dim, batch_size, dropout):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
		#self.word_embeddings = nn.Embedding(len(embeddings), len(embeddings[0]))
		self.lstm = nn.LSTM(len(embeddings[0]), hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
		#self.hidden = self.init_hidden()
		self.dropout = nn.Dropout(dropout)

	#def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
	#	return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
	#			autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		x = embeds.view(len(sentence[0]), self.batch_size, -1)
		#lstm_out, self.hidden = self.lstm(x, self.hidden)
		lstm_out, (hidden, cell) = self.lstm(x)
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
		return lstm_out
		#return embeds

class Feedforward(nn.Module):
	def __init__(self, input_dim, hidden_dim, label_size, batch_size):
		super(Feedforward, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.feedforward = nn.Linear(input_dim, hidden_dim)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.hidden = self.init_hidden()
		self.sigmoid = nn.Sigmoid()

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))

	def forward(self, input):
		out = self.feedforward(input)
		y = self.hidden2label(out)
		probs = self.sigmoid(y)
		return probs	

def get_accuracy(truth, pred):
	assert len(truth)==len(pred)
	right = 0
	for i in range(len(truth)):
		if (truth[i] == 1 and pred[i] >= 0.5) or (truth[i] == 0 and pred[i] < 0.5):
		#if (truth[i] == 1 and pred[i] >= 0.5):
			right += 1.0
	return right/len(truth)

# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, max_len, cuda=False):
	sequences = []
	masks = []
	#mask_idx = len(to_ix)
	mask_idx = 0
	for s in seq:
		sequence = [to_ix[w] if w in to_ix else to_ix['<unk>'] for w in s.split(' ')[:max_len]]
		mask = [1]*len(sequence) + [0]*(max_len - len(sequence))
		masks.append(mask)
		sequence += [mask_idx]*(max_len - len(sequence))
		sequences.append(sequence)
	if cuda:
		sequences = autograd.Variable(torch.LongTensor(sequences).cuda())
		masks = autograd.Variable(torch.FloatTensor(masks).cuda())
	else:
		sequences = autograd.Variable(torch.LongTensor(sequences))
		masks = autograd.Variable(torch.FloatTensor(masks))
	return sequences, masks

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
	return sum(param.numel() for param in model.parameters() if param.requires_grad)

def count_all_parameters(model):
	return sum(param.numel() for param in model.parameters())

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
	dropout = 0.5
	context_model = LSTM(word_embeddings, HIDDEN_DIM, BATCH_SIZE, dropout)
	question_model = LSTM(word_embeddings, HIDDEN_DIM, BATCH_SIZE, dropout)
	answer_model = LSTM(word_embeddings, HIDDEN_DIM, BATCH_SIZE, dropout)
	utility_model = Feedforward(HIDDEN_DIM*2*3, HIDDEN_DIM, 1, batch_size=BATCH_SIZE)

	if args.cuda:
		context_model.cuda()
		question_model.cuda()
		answer_model.cuda()
		utility_model.cuda()

	#loss_function = nn.NLLLoss()
	#loss_function = nn.BCELoss()
	loss_function = nn.BCEWithLogitsLoss()
	print count_grad_parameters(context_model)
	print count_grad_parameters(question_model)
	print count_grad_parameters(answer_model)
	print count_grad_parameters(utility_model)
	print count_all_parameters(context_model)
	print count_all_parameters(question_model)
	print count_all_parameters(answer_model)
	print count_all_parameters(utility_model)
	optimizer = optim.Adam([param for param in itertools.chain(context_model.parameters(), \
											question_model.parameters(), \
											answer_model.parameters(), \
		 									utility_model.parameters()) if param.requires_grad] ,lr = 1e-3)
	no_up = 0
	for i in range(EPOCH):
		random.shuffle(train_data)
		validate(context_model, question_model, answer_model, utility_model, \
						train_data, loss_function, optimizer, vocab, i, BATCH_SIZE, args.cuda, 'train')
		continue
		print('now best dev acc:',best_dev_acc)
		dev_acc = validate(context_model, question_model, answer_model, utility_model, \
						dev_data, loss_function, None, vocab, i, BATCH_SIZE, args.cuda, 'dev')
		#test_acc = validate(context_model, question_model, answer_model, utility_model, \
		#				test_data, loss_function, None, vocab, i, BATCH_SIZE, args.cuda, 'test')
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

def validate(context_model, question_model, answer_model, utility_model, \
					train_data, loss_function, optimizer, vocab, i, batch_size, cuda, split):
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = 0
	batch_sent = []

	if split == 'train':
		context_model.train()
		question_model.train()
		answer_model.train()
		utility_model.train()
	else:
		context_model.eval()
		question_model.eval()
		answer_model.eval()
		utility_model.eval()

	train_data = np.array(train_data)
	train_data = np.transpose(train_data)
	contexts_data, questions_data, answers_data, labels_data = train_data
	for contexts, questions, answers, labels in \
							iterate_minibatches(contexts_data, questions_data, answers_data, labels_data, batch_size):
		labels = prepare_label(labels, cuda)
		# init model
		#context_model.hidden = context_model.init_hidden()
		#question_model.hidden = question_model.init_hidden()
		#answer_model.hidden = answer_model.init_hidden()
		#utility_model.hidden = utility_model.init_hidden()
		# prepare input
		contexts, context_masks = prepare_sequence(contexts, vocab, 250, cuda)
		questions, question_masks = prepare_sequence(questions, vocab, 50, cuda)
		answers, answer_masks = prepare_sequence(answers, vocab, 50, cuda)
		# run LSTM models
		context_out = context_model(contexts)
		question_out = question_model(questions)
		answer_out = answer_model(answers)
		context_masks = torch.transpose(context_masks[:,:,None].expand(context_masks.shape[0], context_masks.shape[1], 200), 0, 1)
		question_masks = torch.transpose(question_masks[:,:,None].expand(question_masks.shape[0], question_masks.shape[1], 200), 0, 1)
		answer_masks = torch.transpose(answer_masks[:,:,None].expand(answer_masks.shape[0], answer_masks.shape[1], 200), 0, 1)
		context_out = torch.mean(context_out * context_masks, 0)
		question_out = torch.mean(question_out * question_masks, 0)
		answer_out = torch.mean(answer_out * answer_masks, 0)
		pred = utility_model(torch.cat((context_out, question_out, answer_out), 1))
		pred = pred[:,0]
		loss = loss_function(pred, labels)
		acc = get_accuracy(labels.data, pred.data)
		avg_loss += loss.item()
		avg_acc += acc
		num_batches += 1
		if split == 'train':
			context_model.zero_grad()
			question_model.zero_grad()
			answer_model.zero_grad()
			utility_model.zero_grad()
			loss.backward()
			optimizer.step()

	avg_loss /= num_batches
	avg_acc /= num_batches
	print('epoch: %d \n %s avg_loss:%g , avg_acc:%g'%(i, split, avg_loss, avg_acc))

	return avg_acc

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


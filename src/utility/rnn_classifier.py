import argparse, sys
import cPickle as p
import numpy as np
import torch
from torchtext import data
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class RNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
		super(RNN, self).__init__()
		
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
		self.fc = nn.Linear(hidden_dim*2, output_dim)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):

		#x = [sent len, batch size]
	  
		embedded = self.dropout(self.embedding(x))
		
		#embedded = [sent len, batch size, emb dim]
		
		output, (hidden, cell) = self.rnn(embedded)
		
		#output = [sent len, batch size, hid dim * num directions]
		#hidden = [num layers * num directions, batch size, hid. dim]
		#cell = [num layers * num directions, batch size, hid. dim]
		
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
				
		#hidden [batch size, hid. dim * num directions]
			
		return self.fc(hidden.squeeze(0))		

import torch.nn.functional as F

def binary_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""

	#round predictions to the closest integer
	rounded_preds = torch.round(F.sigmoid(preds))
	correct = (rounded_preds == y).float() #convert into float for division 
	acc = correct.sum()/len(correct)
	return acc


def train_fn(model, train_data, optimizer, criterion, batch_size):
	
	epoch_loss = 0
	epoch_acc = 0
	
	model.train()
	
	contexts, labels = train_data
	num_batches = 0
	for c, l in iterate_minibatches(contexts, labels, batch_size):
		
		optimizer.zero_grad()
		
		predictions = model(torch.transpose(c, 0, 1)).squeeze(1)
		
		loss = criterion(predictions, l)
		
		acc = binary_accuracy(predictions, l)
		
		loss.backward()
		
		optimizer.step()
		
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		num_batches += 1
		
	return epoch_loss / num_batches, epoch_acc / num_batches

def evaluate(model, dev_data, criterion, batch_size):
	
	epoch_loss = 0
	epoch_acc = 0
	num_batches = 0	
	model.eval()
	
	with torch.no_grad():
	
		contexts, labels = dev_data
		for c, l in iterate_minibatches(contexts, labels, batch_size):

			predictions = model(torch.transpose(c, 0, 1)).squeeze(1)
			
			loss = criterion(predictions, l)
			
			acc = binary_accuracy(predictions, l)

			epoch_loss += loss.item()
			epoch_acc += acc.item()
			num_batches += 1
		
	return epoch_loss / num_batches, epoch_acc / num_batches

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
	#if cuda:
	#	sequences = autograd.Variable(torch.LongTensor(sequences).cuda())
	#	masks = autograd.Variable(torch.FloatTensor(masks).cuda())
	#else:
	#	sequences = autograd.Variable(torch.LongTensor(sequences))
	#	masks = autograd.Variable(torch.FloatTensor(masks))
	return sequences, masks

def prepare_data(input_data, vocab, split, cuda):
	input_data = np.array(input_data)
	input_data = np.transpose(input_data)
	contexts_data, questions_data, answers_data, labels_data = input_data
	#print len(contexts_data)
	contexts_data = contexts_data[:80240]
	questions_data = questions_data[:80240]
	answers_data = answers_data[:80240]
	labels_data = np.array(labels_data, dtype=np.int32)
	labels_data = labels_data[:80240]
	#print np.count_nonzero(labels_data)
	contexts, context_masks = prepare_sequence(contexts_data, vocab, 250, cuda)
	questions, question_masks = prepare_sequence(questions_data, vocab, 50, cuda)
	answers, answer_masks = prepare_sequence(answers_data, vocab, 50, cuda)
	labels = prepare_label(labels_data, cuda)
	output_data = [contexts, context_masks, questions, question_masks, answers, answer_masks, labels]
	return output_data	

def prepare_label(labels, cuda=False):
	if cuda:
		#var = autograd.Variable(torch.FloatTensor([float(l) for l in labels]).cuda())
		var = torch.FloatTensor([float(l) for l in labels]).cuda()
	else:
		var = autograd.Variable(torch.FloatTensor([float(l) for l in labels]))
	return var

def iterate_minibatches(contexts, labels, batch_size, shuffle=True):
	if shuffle:
		indices = np.arange(len(contexts))
		np.random.shuffle(indices)
	for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield contexts[excerpt], labels[excerpt]

def main(args):
	SEED = 1234

	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)

	train_data = p.load(open(args.train_data, 'rb'))
	dev_data = p.load(open(args.tune_data, 'rb'))
	test_data = p.load(open(args.test_data, 'rb'))
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	vocab = p.load(open(args.vocab, 'rb'))

	BATCH_SIZE = 64
	INPUT_DIM = len(vocab)
	EMBEDDING_DIM = 200
	HIDDEN_DIM = 256
	OUTPUT_DIM = 1
	N_LAYERS = 2
	BIDIRECTIONAL = True
	DROPOUT = 0.5

	model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

	word_embeddings = autograd.Variable(torch.FloatTensor(word_embeddings).cuda())
	model.embedding.weight.data.copy_(word_embeddings)
	optimizer = optim.Adam(model.parameters())
	criterion = nn.BCEWithLogitsLoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = criterion.to(device)

	N_EPOCHS = 300
	train_data = prepare_data(train_data, vocab, 'train', args.cuda)
	dev_data = prepare_data(dev_data, vocab, 'dev', args.cuda)
	test_data = prepare_data(test_data, vocab, 'test', args.cuda)

	posts_data, post_masks_data, \
	questions_data, question_masks_data, \
	answers_data, answer_masks_data, labels_data = train_data

	contexts_data = [None]*len(posts_data)
	for i in range(len(posts_data)):
		contexts_data[i] = posts_data[i][:50] + questions_data[i][:50]
	#contexts_data = autograd.Variable(torch.LongTensor(contexts_data).cuda())
	contexts_data = torch.LongTensor(contexts_data).cuda()
	new_train_data = [contexts_data, labels_data]	

	posts_data, post_masks_data, \
	questions_data, question_masks_data, \
	answers_data, answer_masks_data, labels_data = dev_data

	contexts_data = [None]*len(posts_data)
	for i in range(len(posts_data)):
		contexts_data[i] = posts_data[i][:50] + questions_data[i][:50]
	#contexts_data = autograd.Variable(torch.LongTensor(contexts_data).cuda())
	contexts_data = torch.LongTensor(contexts_data).cuda()
	new_dev_data = [contexts_data, labels_data]	

	for epoch in range(N_EPOCHS):
		train_loss, train_acc = train_fn(model, new_train_data, optimizer, criterion, BATCH_SIZE)
		valid_loss, valid_acc = evaluate(model, new_dev_data, criterion, BATCH_SIZE)
		print 'Epoch %d: Train Loss: %.3f, Train Acc: %.3f, Val Loss: %.3f, Val Acc: %.3f' % (epoch, train_loss, train_acc, valid_loss, valid_acc)   

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--train_data", type = str)
	argparser.add_argument("--tune_data", type = str)
	argparser.add_argument("--test_data", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--cuda", type = bool)
	args = argparser.parse_args()
	print args
	print ""
	main(args)
 

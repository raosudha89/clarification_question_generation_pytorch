import argparse, sys
import pickle as p
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

class FeedForward(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(FeedForward, self).__init__()
		self.layer1 = nn.Linear(input_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.layer2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		return x

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


def train_fn(context_model, question_model, answer_model, utility_model, train_data, optimizer, criterion, batch_size):
	
	epoch_loss = 0
	epoch_acc = 0
	
	context_model.train()
	question_model.train()
	answer_model.train()
	utility_model.train()
	
	contexts, questions, answers, labels = train_data
	num_batches = 0
	for c, q, a, l in iterate_minibatches(contexts, questions, answers, labels, batch_size):
		
		optimizer.zero_grad()
		
		c_out = context_model(torch.transpose(c, 0, 1)).squeeze(1)
		q_out = question_model(torch.transpose(q, 0, 1)).squeeze(1)
		a_out = answer_model(torch.transpose(a, 0, 1)).squeeze(1)
		predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
		
		loss = criterion(predictions, l)
		
		acc = binary_accuracy(predictions, l)
		
		loss.backward()
		
		optimizer.step()
		
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		num_batches += 1
		
	return epoch_loss / num_batches, epoch_acc / num_batches

def evaluate(context_model, question_model, answer_model, utility_model, dev_data, criterion, batch_size):
	
	epoch_loss = 0
	epoch_acc = 0
	num_batches = 0	

	context_model.eval()
	question_model.eval()
	answer_model.eval()
	utility_model.eval()
	
	with torch.no_grad():
	
		contexts, questions, answers, labels = dev_data
		for c, q, a, l in iterate_minibatches(contexts, questions, answers, labels, batch_size):

			c_out = context_model(torch.transpose(c, 0, 1)).squeeze(1)
			q_out = question_model(torch.transpose(q, 0, 1)).squeeze(1)
			a_out = answer_model(torch.transpose(a, 0, 1)).squeeze(1)
			predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
			
			loss = criterion(predictions, l)
			
			acc = binary_accuracy(predictions, l)

			epoch_loss += loss.item()
			epoch_acc += acc.item()
			num_batches += 1
		
	return epoch_loss / num_batches, epoch_acc / num_batches

def prepare_sequence(seq, to_ix, max_len, cuda=False):
	sequences = []
	mask_idx = 0
	for s in seq:
		sequence = [to_ix[w] if w in to_ix else to_ix['<unk>'] for w in s.split(' ')[:max_len]]
		sequence += [mask_idx]*int(max_len - len(sequence))
		sequences.append(sequence)
	sequences = torch.LongTensor(sequences).cuda()
	return sequences

def prepare_data(input_data, vocab, split, cuda):
	input_data = np.array(input_data)
	input_data = np.transpose(input_data)
	contexts_data, questions_data, answers_data, labels_data = input_data
	#contexts_data = contexts_data[:80240]
	#questions_data = questions_data[:80240]
	#answers_data = answers_data[:80240]
	#labels_data = labels_data[:80240]
	labels_data = np.array(labels_data, dtype=np.int32)
	contexts = prepare_sequence(contexts_data, vocab, 250, cuda)
	questions = prepare_sequence(questions_data, vocab, 50, cuda)
	answers = prepare_sequence(answers_data, vocab, 50, cuda)
	labels = prepare_label(labels_data, cuda)
	output_data = [contexts, questions, answers, labels]
	return output_data	

def prepare_label(labels, cuda=False):
	if cuda:
		#var = autograd.Variable(torch.FloatTensor([float(l) for l in labels]).cuda())
		var = torch.FloatTensor([float(l) for l in labels]).cuda()
	else:
		var = autograd.Variable(torch.FloatTensor([float(l) for l in labels]))
	return var

def iterate_minibatches(contexts, questions, answers, labels, batch_size, shuffle=True):
	if shuffle:
		indices = np.arange(len(contexts))
		np.random.shuffle(indices)
	for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield contexts[excerpt], questions[excerpt], answers[excerpt], labels[excerpt]

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
	EMBEDDING_DIM = len(word_embeddings[0])
	HIDDEN_DIM = 100
	OUTPUT_DIM = 1
	N_LAYERS = 1
	BIDIRECTIONAL = True
	DROPOUT = 0.5

	context_model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
	question_model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
	answer_model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

	utility_model = FeedForward(HIDDEN_DIM*3, HIDDEN_DIM, OUTPUT_DIM)

	criterion = nn.BCEWithLogitsLoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	utility_model = utility_model.to(device)
	context_model = context_model.to(device)
	question_model = question_model.to(device)
	answer_model = answer_model.to(device)
	criterion = criterion.to(device)

	word_embeddings = autograd.Variable(torch.FloatTensor(word_embeddings).cuda())
	context_model.embedding.weight.data.copy_(word_embeddings)
	question_model.embedding.weight.data.copy_(word_embeddings)
	answer_model.embedding.weight.data.copy_(word_embeddings)

	# Fix word embeddings
	context_model.embedding.weight.requires_grad = False
	question_model.embedding.weight.requires_grad = False
	answer_model.embedding.weight.requires_grad = False

	optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) + \
							list([par for par in question_model.parameters() if par.requires_grad]) + \
							list([par for par in answer_model.parameters() if par.requires_grad]) + \
							list([par for par in utility_model.parameters() if par.requires_grad]))


	N_EPOCHS = 300
	train_data = prepare_data(train_data, vocab, 'train', args.cuda)
	dev_data = prepare_data(dev_data, vocab, 'dev', args.cuda)
	test_data = prepare_data(test_data, vocab, 'test', args.cuda)

	for epoch in range(N_EPOCHS):
		train_loss, train_acc = train_fn(context_model, question_model, answer_model, utility_model, \
											train_data, optimizer, criterion, BATCH_SIZE)
		valid_loss, valid_acc = evaluate(context_model, question_model, answer_model, utility_model, \
                                            dev_data, criterion, BATCH_SIZE)
		#valid_loss, valid_acc = evaluate(context_model, question_model, answer_model, utility_model, \
        #                                    test_data, criterion, BATCH_SIZE)
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
 

from helper import *
import numpy as np
import torch
from RL_constants import *

def train_fn(context_model, question_model, answer_model, utility_model, train_data, optimizer, criterion):
	epoch_loss = 0
	epoch_acc = 0
	
	context_model.train()
	question_model.train()
	answer_model.train()
	utility_model.train()
	
	contexts, questions, answers, labels = train_data
	contexts = np.array(contexts)
	questions = np.array(questions)
	answers = np.array(answers)
	labels = np.array(labels)

	num_batches = 0
	for c, q, a, l in iterate_minibatches(contexts, questions, answers, labels, batch_size):
		optimizer.zero_grad()
		c_out = context_model(torch.transpose(torch.tensor(c).cuda(), 0, 1)).squeeze(1)
		q_out = question_model(torch.transpose(torch.tensor(q).cuda(), 0, 1)).squeeze(1)
		a_out = answer_model(torch.transpose(torch.tensor(a).cuda(), 0, 1)).squeeze(1)
		predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
		l = torch.FloatTensor([float(lab) for lab in l]).cuda()
		loss = criterion(predictions, l)
		acc = binary_accuracy(predictions, l)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		num_batches += 1
		
	return epoch_loss / num_batches, epoch_acc / num_batches



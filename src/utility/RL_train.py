from helper import *
import numpy as np
import torch
from constants import *

def train_utility(context_model, question_model, answer_model, utility_model, optimizer, c, q, a):
	context_model.train()
	question_model.train()
	answer_model.train()
	utility_model.train()
	optimizer.zero_grad()
	criterion = torch.nn.BCEWithLogitsLoss()
	if USE_CUDA:
		c_out = context_model(torch.transpose(torch.tensor(c).cuda(), 0, 1)).squeeze(1)
		q_out = question_model(torch.transpose(torch.tensor(q).cuda(), 0, 1)).squeeze(1)
		a_out = answer_model(torch.transpose(torch.tensor(a).cuda(), 0, 1)).squeeze(1)
	else:
		c_out = context_model(torch.transpose(torch.tensor(c), 0, 1)).squeeze(1)
		q_out = question_model(torch.transpose(torch.tensor(q), 0, 1)).squeeze(1)
		a_out = answer_model(torch.transpose(torch.tensor(a), 0, 1)).squeeze(1)
	predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
	l = torch.FloatTensor(np.ones(len(predictions)))
	if USE_CUDA:
		l = l.cuda()
	loss = criterion(predictions, l)
	return loss, predictions	

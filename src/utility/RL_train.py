from helper import *
import numpy as np
import torch
from constants import *

def train_utility(context_model, question_model, answer_model, utility_model, optimizer, c, cl, q, ql, a, al, args, all_pos=True):
	context_model.train()
	question_model.train()
	answer_model.train()
	utility_model.train()
	optimizer.zero_grad()
	criterion = torch.nn.BCEWithLogitsLoss()
	cm = get_masks(cl, args.max_post_len)
	qm = get_masks(ql, args.max_ques_len)
	am = get_masks(al, args.max_ans_len)
	c = torch.tensor(c)
	cm = torch.FloatTensor(cm)
	q = torch.tensor(q)
	qm = torch.FloatTensor(qm)
	a = torch.tensor(a)
	am = torch.FloatTensor(am)
	if USE_CUDA:
		c = c.cuda()
		cm = cm.cuda()
		q = q.cuda()
		qm = qm.cuda()
		a = a.cuda()
		am = am.cuda()
	c_hid, c_out = context_model(torch.transpose(c, 0, 1))
	c_hid = c_hid.squeeze(1)
	cm = torch.transpose(cm, 0, 1).unsqueeze(2)
	cm = cm.expand(cm.shape[0], cm.shape[1], 2*HIDDEN_SIZE)
	c_out = torch.sum(c_out * cm, dim=0)

	q_hid, q_out = question_model(torch.transpose(q, 0, 1))
	q_hid = q_hid.squeeze(1)
	qm = torch.transpose(qm, 0, 1).unsqueeze(2)
	qm = qm.expand(qm.shape[0], qm.shape[1], 2*HIDDEN_SIZE)
	q_out = torch.sum(q_out * qm, dim=0)

	a_hid, a_out = answer_model(torch.transpose(q, 0, 1))
	a_hid = a_hid.squeeze(1)
	am = torch.transpose(am, 0, 1).unsqueeze(2)
	am = am.expand(am.shape[0], am.shape[1], 2*HIDDEN_SIZE)
	a_out = torch.sum(a_out * am, dim=0)

	predictions = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
	if all_pos:
		l = torch.FloatTensor(np.ones(len(predictions)))
	else:
		l = torch.FloatTensor(np.zeros(len(predictions)))
	if USE_CUDA:
		l = l.cuda()
	loss = criterion(predictions, l)
	return loss, predictions	

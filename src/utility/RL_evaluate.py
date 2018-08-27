from constants import *
from helper import *
import numpy as np
import torch
from torch.autograd import Variable

def evaluate_utility(context_model, question_model, answer_model, utility_model, c, q, a):
	with torch.no_grad():
		context_model.eval()
		question_model.eval()
		answer_model.eval()
		utility_model.eval()
		if USE_CUDA:
			c =  Variable(torch.LongTensor(c).cuda().transpose(0, 1))
			q =  Variable(torch.LongTensor(q).cuda().transpose(0, 1))
			a =  Variable(torch.LongTensor(a).cuda().transpose(0, 1))
		else:
			c =  Variable(torch.LongTensor(c).transpose(0, 1))
			q =  Variable(torch.LongTensor(q).transpose(0, 1))
			a =  Variable(torch.LongTensor(a).transpose(0, 1))
		c_out = context_model(c).squeeze(1)
		q_out = context_model(q).squeeze(1)
		a_out = context_model(a).squeeze(1)
		preds = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
		preds = torch.nn.functional.sigmoid(preds)
		del c, q, a
	return preds	

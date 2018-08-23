from helper import *
import numpy as np
import torch
from constants import *

def evaluate(context_model, question_model, answer_model, utility_model, c, q, a):
	context_model.eval()
	question_model.eval()
	answer_model.eval()
	utility_model.eval()
	c_out = context_model(torch.transpose(torch.tensor(c).cuda(), 0, 1)).squeeze(1)
	q_out = question_model(torch.transpose(torch.tensor(q).cuda(), 0, 1)).squeeze(1)
	a_out = answer_model(torch.transpose(torch.tensor(a).cuda(), 0, 1)).squeeze(1)
	preds = utility_model(torch.cat((c_out, q_out, a_out), 1)).squeeze(1)
	preds = torch.nn.functional.sigmoid(preds)
	return preds	

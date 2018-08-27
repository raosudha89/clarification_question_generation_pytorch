from constants import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attn(nn.Module):
	def __init__(self, hidden_size):
		super(Attn, self).__init__()
		self.hidden_size = hidden_size
		
	def forward(self, hidden, encoder_outputs):
		max_len = encoder_outputs.size(0)
		this_batch_size = encoder_outputs.size(1)

		# Create variable to store attention energies
		attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
		if USE_CUDA:
			attn_energies = attn_energies.cuda()
		attn_energies = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(0,1).transpose(1,2)).squeeze(1)
		# Normalize energies to weights in range 0 to 1, resize to 1 x B x S
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

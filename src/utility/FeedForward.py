import torch
import torch.nn as nn
from constants import *

class FeedForward(nn.Module):
	def __init__(self, input_dim):
		super(FeedForward, self).__init__()

		self.layer1 = nn.Linear(input_dim, HIDDEN_SIZE)
		self.relu = nn.ReLU()
		self.layer2 = nn.Linear(HIDDEN_SIZE, 1)

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		return x


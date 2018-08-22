import torch
import torch.nn as nn
from RL_constants import *

class RNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim):
		super(RNN, self).__init__()
		
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, bidirectional=True, dropout=dropout)
		self.fc = nn.Linear(hidden_size*2, hidden_size)
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


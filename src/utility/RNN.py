import torch
import torch.nn as nn
from constants import *


class RNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n_layers):
		super(RNN, self).__init__()
		
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.LSTM(embedding_dim, HIDDEN_SIZE, num_layers=n_layers, bidirectional=True)
		self.fc = nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE)
		self.dropout = nn.Dropout(DROPOUT)
		
	def forward(self, x):

		#x = [sent len, batch size]

		embedded = self.dropout(self.embedding(x))
		
		#embedded = [sent len, batch size, emb dim]
		
		output, (hidden, cell) = self.rnn(embedded)
		
		#output = [sent len, batch size, hid dim * num directions]
		#hidden = [num layers * num directions, batch size, hid. dim]
		#cell = [num layers * num directions, batch size, hid. dim]
		
		hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
				
		#hidden [batch size, hid. dim * num directions]
			
		return self.fc(hidden.squeeze(0)), output	


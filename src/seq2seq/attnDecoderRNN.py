import torch
import torch.nn as nn
from attn import *


class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, word_embeddings, n_layers=1, dropout=0.1):
		super(AttnDecoderRNN, self).__init__()

		# Keep for reference
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
		self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
		self.embedding.weight.requires_grad = False
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(len(word_embeddings[0]), hidden_size, n_layers, dropout=dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		
		self.attn = Attn(hidden_size)

	def forward(self, input_seq, last_hidden, p_encoder_outputs):
		# Note: we run this one step at a time

		# Get the embedding of the current input word (last output word)
		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, embedded.shape[0], embedded.shape[1]) # S=1 x B x N

		# Get current hidden state from input word and last hidden state
		rnn_output, hidden = self.gru(embedded, last_hidden)

		# Calculate attention from current RNN state and all p_encoder outputs;
		# apply to p_encoder outputs to get weighted average
		p_attn_weights = self.attn(rnn_output, p_encoder_outputs)
		p_context = p_attn_weights.bmm(p_encoder_outputs.transpose(0, 1)) # B x S=1 x N

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N

		p_context = p_context.squeeze(1)	   # B x S=1 x N -> B x N
		concat_input = torch.cat((rnn_output, p_context), 1)
		concat_output = F.tanh(self.concat(concat_input))

		# Finally predict next token (Luong eq. 6, without softmax)
		output = self.out(concat_output)

		# Return final output, hidden state, and attention weights (for visualization)
		return output, hidden

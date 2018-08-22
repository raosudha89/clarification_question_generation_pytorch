import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
	def __init__(self, hidden_size, word_embeddings, n_layers=1, dropout=0.1):
		super(EncoderRNN, self).__init__()
		
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout
		
		self.embedding = nn.Embedding(len(word_embeddings), len(word_embeddings[0]))
		self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
		self.embedding.weight.requires_grad = False
		self.gru = nn.GRU(len(word_embeddings[0]), hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
		
	def forward(self, input_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding(input_seqs)
		#packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		#outputs, hidden = self.gru(packed, hidden)
		#outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
		outputs, hidden = self.gru(embedded, hidden)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
		return outputs, hidden

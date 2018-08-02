import torch
import torch.nn as nn

class EncoderAvgEmb(nn.Module):
	def __init__(self, pretrained_emb):
		super(EncoderAvgEmb, self).__init__()
		
		self.input_size = pretrained_emb.shape[0]
		self.hidden_size = pretrained_emb.shape[1]
		
		self.embedding = nn.Embedding(self.input_size, self.hidden_size)
		self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
		#self.embedding.weight = nn.Parameter(pretrained_emb)
		
	def forward(self, input_seqs):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding(input_seqs)
		output = torch.mean(embedded, dim=0)
		return output

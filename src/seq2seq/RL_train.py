import torch
from torch.autograd import Variable
from masked_cross_entropy import *
import numpy as np
from RL_constants import *

def train(input_batches, input_lens, target_batches, target_lens, \
			encoder, decoder, encoder_optimizer, decoder_optimizer, SOS_idx):
	
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	if USE_CUDA:
		input_batches = Variable(torch.LongTensor(np.array(input_batches)).cuda()).transpose(0, 1)
		target_batches = Variable(torch.LongTensor(np.array(target_batches)).cuda()).transpose(0, 1)

	# Run post words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

	# Prepare input and output variables
	decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
	max_target_length = 50

	decoder_input = Variable(torch.LongTensor([SOS_idx] * batch_size).cuda())
	all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size).cuda())

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
		all_decoder_outputs[t] = decoder_output

		# Teacher Forcing
		decoder_input = target_batches[t] # Next input is current target

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_batches.transpose(0, 1).contiguous(), # -> batch x seq
		target_lens
	)
	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()
	return loss.item()

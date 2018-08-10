import torch
from torch.autograd import Variable
from masked_cross_entropy import *
import numpy as np
from constants import *

def train(input_batches, input_lengths, target_batches, target_lengths, \
			encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, USE_CUDA):
	
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0 # Added onto for each word

	# Run post words through encoder
	#import pdb
	#pdb.set_trace()
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

	# Prepare input and output variables
	decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
	#max_target_length = max(target_lengths)
	max_target_length = 50

	# Move new Variables to CUDA
	if USE_CUDA:
		decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size).cuda())
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size).cuda())
	else:
		decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

		all_decoder_outputs[t] = decoder_output
		decoder_input = target_batches[t] # Next input is current target

		# Get most likely word index (highest value) from output
		#topv, topi = decoder_output.data.topk(1)
		#ni = topi[0][0]

		#decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
		#if USE_CUDA: decoder_input = decoder_input.cuda()

		# Stop at end of sentence 
		#if ni == EOS_token: break		

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_batches.transpose(0, 1).contiguous(), # -> batch x seq
		target_lengths
	)
	loss.backward()
	
	# Clip gradient norms
	ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

	# Update parameters with optimizers
	encoder_optimizer.step()
	decoder_optimizer.step()
	
	return loss.data[0], ec, dc

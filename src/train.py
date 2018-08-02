import torch
from torch.autograd import Variable
from masked_cross_entropy import *
import numpy as np
from constants import *

def train(p_input_batches, p_input_lengths, q_input_batches_list, q_input_lengths_list, target_batches, target_lengths, \
			p_encoder, q_encoder, decoder, p_encoder_optimizer, q_encoder_optimizer, decoder_optimizer, criterion, USE_CUDA):
	
	# Zero gradients of both optimizers
	p_encoder_optimizer.zero_grad()
	q_encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0 # Added onto for each word

	# Run post words through p_encoder
	p_encoder_outputs, p_encoder_hidden = p_encoder(p_input_batches, p_input_lengths, None)

	# Run words through encoder
	N_q = len(q_input_batches_list)
	q_encoder_outputs_list, q_encoder_hidden_list = [None]*N_q, [None]*N_q
	for i in range(N_q):
		q_encoder_outputs_list[i], q_encoder_hidden_list[i] = q_encoder(q_input_batches_list[i], q_input_lengths_list[i], None)
	
	# Prepare input and output variables
	#decoder_hidden = p_encoder_output + encoder_hidden[:decoder.n_layers] # Use avg p emb + last (forward) hidden state from encoder
	#decoder_hidden = p_encoder_hidden[:decoder.n_layers] + q_encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
	decoder_hidden = p_encoder_hidden[:decoder.n_layers] + p_encoder_hidden[decoder.n_layers:]
	for i in range(N_q):
		decoder_hidden += (q_encoder_hidden_list[i][:decoder.n_layers] + q_encoder_hidden_list[i][decoder.n_layers:])

	max_target_length = max(target_lengths)

	# Move new Variables to CUDA
	if USE_CUDA:
		decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size).cuda())
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size).cuda())
	else:
		decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

	# Run through decoder one time step at a time
	#print 'max target len %d ' % max_target_length
	for t in range(max_target_length):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, \
												p_encoder_outputs, q_encoder_outputs_list)

		all_decoder_outputs[t] = decoder_output
		decoder_input = target_batches[t] # Next input is current target

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_batches.transpose(0, 1).contiguous(), # -> batch x seq
		target_lengths
	)
	loss.backward()
	
	# Clip gradient norms
	ec = torch.nn.utils.clip_grad_norm(p_encoder.parameters(), clip)
	ec = torch.nn.utils.clip_grad_norm(q_encoder.parameters(), clip)
	dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

	# Update parameters with optimizers
	p_encoder_optimizer.step()
	q_encoder_optimizer.step()
	decoder_optimizer.step()
	
	return loss.data[0], ec, dc

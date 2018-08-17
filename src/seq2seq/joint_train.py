import torch
from torch.autograd import Variable
from masked_cross_entropy import *
import numpy as np
from constants import *

def train_seq2seq(input_batches, input_lens, target_batches, target_lens, \
			encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, USE_CUDA):
	
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	# Run post words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

	# Prepare input and output variables
	decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
	max_target_length = 50

	decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size).cuda())
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
	return encoder_optimizer, decoder_optimizer, loss

def train(post_batches, post_lens, ques_batches, ques_lens, ans_batches, ans_lens, \
			q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer, q_criterion, \
			a_encoder, a_decoder, a_encoder_optimizer, a_decoder_optimizer, a_criterion, USE_CUDA):

	q_encoder_optimizer, q_decoder_optimizer, q_loss = train_seq2seq(post_batches, post_lens, \
															ques_batches, ques_lens, \
															q_encoder, q_decoder, \
															q_encoder_optimizer, q_decoder_optimizer, \
															q_criterion, USE_CUDA)													 

	a_encoder_optimizer, a_decoder_optimizer, a_loss = train_seq2seq(ques_batches, ques_lens, \
															ans_batches, ans_lens, \
															a_encoder, a_decoder, \
															a_encoder_optimizer, a_decoder_optimizer, \
															a_criterion, USE_CUDA)															 
	
	loss = q_loss + a_loss
	loss.backward()
	
	# Update parameters with optimizers
	q_encoder_optimizer.step()
	q_decoder_optimizer.step()
	a_encoder_optimizer.step()
	a_decoder_optimizer.step()
	
	return loss.item()

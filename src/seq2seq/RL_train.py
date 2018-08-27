from constants import *
from masked_cross_entropy import *
import numpy as np
import random
import torch
from torch.autograd import Variable

def get_decoded_seqs(decoder_outputs, word2index, max_len, batch_size):
	decoded_seqs_N = []
	decoded_lens_N = []
	decoded_seq_masks_N = []
	N = 10
	for i in range(N):
		decoded_seqs = []
		decoded_lens = []
		decoded_seq_masks = []
		for b in range(batch_size):
			decoded_seq = []
			decoded_seq_mask = [0]*max_len
			log_prob = 0.
			for t in range(max_len):
				topi = decoder_outputs[i][t][b].data.topk(N)[1][i]
				ni = topi.item()
				if ni == word2index[EOS_token]:
					decoded_seq.append(ni)
					break
				else:
					decoded_seq.append(ni)
					decoded_seq_mask[t] = 1
			decoded_lens.append(len(decoded_seq))
			decoded_seq += [word2index[PAD_token]]*(max_len - len(decoded_seq))
			decoded_seqs.append(decoded_seq)
			decoded_seq_masks.append(decoded_seq_mask)
		decoded_seqs_N.append(decoded_seqs)
		decoded_lens_N.append(decoded_lens)
		decoded_seq_masks_N.append(decoded_seq_masks)

	decoded_lens_N = np.array(decoded_lens_N)
	decoded_seqs_N = np.array(decoded_seqs_N)
	decoded_seq_masks_N = np.array(decoded_seq_masks_N)
	return decoded_seqs_N, decoded_lens_N, decoded_seq_masks_N

def train(input_batches, input_lens, target_batches, target_lens, \
			encoder, decoder, encoder_optimizer, decoder_optimizer, word2index, max_len, batch_size):
	
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	encoder.train()
	decoder.train()

	N = 10
	input_batches = Variable(torch.LongTensor(np.array(input_batches))).transpose(0, 1)
	target_batches = Variable(torch.LongTensor(np.array(target_batches))).transpose(0, 1)

	decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size))
	decoder_outputs = Variable(torch.zeros(N, max_len, batch_size, decoder.output_size))

	if USE_CUDA:
		input_batches = input_batches.cuda()
		target_batches = target_batches.cuda()
		decoder_input = decoder_input.cuda()
		decoder_outputs = decoder_outputs.cuda()

	# Run post words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

	# Prepare input and output variables
	decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

	# Run through decoder one time step at a time
	for i in range(N):
		for t in range(max_len):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_outputs[i][t] = decoder_output

			# Teacher Forcing
			#decoder_input = target_batches[t] # Next input is current target

			# Greeding decoding
			for b in range(batch_size):
				topi = decoder_output[b].topk(N)[1][i]	
				decoder_input[b] = topi.squeeze().detach()

	decoded_seqs, decoded_lens, decoded_seq_masks = get_decoded_seqs(decoder_outputs, word2index, max_len, batch_size) 

	# Loss calculation and backpropagation
	#loss = masked_cross_entropy(
	#	decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
	#	target_batches.transpose(0, 1).contiguous(), # -> batch x seq
	#	target_lens
	#)

	log_probs = [None]*N
	for i in range(N):
		log_probs[i] = calculate_log_probs(
			decoder_outputs[i].transpose(0, 1).contiguous(), # -> batch x seq
			decoded_seq_masks[i], idx=i
		)

	return log_probs, decoded_seqs, decoded_lens

import torch
from torch.autograd import Variable
from masked_cross_entropy import *
import numpy as np
from constants import *

def train(input_batches, input_lens, target_batches, target_lens, \
			encoder, decoder, encoder_optimizer, decoder_optimizer, word2index, max_len):
	
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

	decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * BATCH_SIZE).cuda())
	decoder_outputs = Variable(torch.zeros(max_target_length, BATCH_SIZE, decoder.output_size).cuda())

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
		decoder_outputs[t] = decoder_output

		# Teacher Forcing
		decoder_input = target_batches[t] # Next input is current target

	decoded_seqs = []
	decoded_lens = []
	for b in range(BATCH_SIZE):
		decoded_seq = []
		log_prob = 0.
		for t in range(max_len):
			topv, topi = decoder_outputs[t][b].data.topk(1)
			ni = topi[0].item()
			if ni == word2index[EOS_token]:
				decoded_seq.append(ni)
				break
			else:
				decoded_seq.append(ni)
		decoded_lens.append(len(decoded_seq))
		decoded_seq += [word2index[PAD_token]]*(max_len - len(decoded_seq))
		decoded_seqs.append(decoded_seq)

	decoded_lens = np.array(decoded_lens)
	decoded_seqs = np.array(decoded_seqs)

	log_probs = calculate_log_probs(
		decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_lens
	)

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_batches.transpose(0, 1).contiguous(), # -> batch x seq
		target_lens
	)

	return loss, log_probs, decoded_seqs, decoded_lens 

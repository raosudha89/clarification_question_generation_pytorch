from constants import *
from masked_cross_entropy import *
import numpy as np
import random
import torch
from torch.autograd import Variable

def get_decoded_seqs(decoder_outputs, word2index, max_len, batch_size):
	decoded_seqs = []
	decoded_lens = []
	decoded_seq_masks = []
	for b in range(batch_size):
		decoded_seq = []
		decoded_seq_mask = [0]*max_len
		log_prob = 0.
		for t in range(max_len):
			topv, topi = decoder_outputs[t][b].data.topk(1)
			ni = topi[0].item()
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
	decoded_lens = np.array(decoded_lens)
	decoded_seqs = np.array(decoded_seqs)
	return decoded_seqs, decoded_lens, decoded_seq_masks

def inference(input_batches, input_lens, target_batches, target_lens, \
			encoder, decoder, word2index, max_len, batch_size):
	
	input_batches = Variable(torch.LongTensor(np.array(input_batches))).transpose(0, 1)
	target_batches = Variable(torch.LongTensor(np.array(target_batches))).transpose(0, 1)

	decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size))
	decoder_outputs = Variable(torch.zeros(max_len, batch_size, decoder.output_size))

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
	for t in range(max_len):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
		decoder_outputs[t] = decoder_output

		# Without teacher forcing
		for b in range(batch_size):
			topv, topi = decoder_output[b].topk(1)	
			decoder_input[b] = topi.squeeze().detach()

	decoded_seqs, decoded_lens, decoded_seq_masks = get_decoded_seqs(decoder_outputs, word2index, max_len, batch_size) 

	# Loss calculation and backpropagation
	#loss = masked_cross_entropy(
	#	decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
	#	target_batches.transpose(0, 1).contiguous(), # -> batch x seq
	#	target_lens
	#)

	log_probs = calculate_log_probs(
		decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		decoded_seq_masks,
	)

	return log_probs, decoded_seqs, decoded_lens

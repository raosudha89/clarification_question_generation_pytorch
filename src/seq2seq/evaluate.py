import random
from constants import *
from prepare_data import indexes_from_sentence
import torch
import torch.nn as nn
from torch.autograd import Variable

def evaluate_randomly(p_data, q_data, triples, p_encoder, decoder):
	[input_sentence, target_sentence] = random.choice(triples)
	evaluate_and_show_attention(p_data, q_data, p_encoder, decoder, input_sentence, target_sentence)
	
def evaluate_and_show_attention(p_data, q_data, p_encoder, decoder, input_sentence, target_sentence=None):
	output_words = evaluate(p_data, q_data, p_encoder, decoder, input_sentence)
	output_sentence = ' '.join(output_words)
	print('>', input_sentence)
	if target_sentence is not None:
		print('=', target_sentence)
	print('<', output_sentence)

def evaluate_testset(p_data, q_data, p_encoder, decoder, test_triples, batch_size):
	input_seqs, target_seqs = preprocess_data(p_data, q_data, test_triples)
	print_loss_total = 0
	for input_seqs_batch, target_seqs_batch in iterate_minibatches(input_seqs, target_seqs, batch_size):
		input_batch, input_lens, target_batch, target_lens = add_padding(input_seqs_batch, target_seqs_batch, USE_CUDA)
		if USE_CUDA:
			input_batches = input_batches.cuda()

		# Run post words through p_encoder
		p_encoder_outputs, p_encoder_hidden = p_encoder(input_batches, input_lengths, None)

		# Create starting vectors for decoder
		decoder_input = Variable(torch.LongTensor([SOS_token * batch_size]), volatile=True)	
		decoder_hidden = p_encoder_hidden[:decoder.n_layers] + p_encoder_hidden[decoder.n_layers:]
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

		if USE_CUDA:
			decoder_input = decoder_input.cuda()
			all_decoder_outputs = all_decoder_outputs.cuda()

		decoded_words = []

		# Run through decoder one time step at a time
		for t in range(max_target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

			# Choose top word from output
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0].item()
			if ni == EOS_token:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(q_data.index2word[ni])	

			all_decoder_outputs[t] = decoder_output
			decoder_input = Variable(torch.LongTensor([ni])) 
			if USE_CUDA: decoder_input = decoder_input.cuda()

		# Loss calculation
		loss = masked_cross_entropy(
			all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
			target_batches.transpose(0, 1).contiguous(), # -> batch x seq
			target_lengths
			)

		print_total_loss += loss.data[0]

def evaluate(p_data, q_data, p_encoder, decoder, input_seq, max_length=MAX_POST_LEN):
	input_seqs = [indexes_from_sentence(p_data, input_seq)]
	input_lengths = [len(input_seqs)]
	input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
	
	if USE_CUDA:
		input_batches = input_batches.cuda()
		
	# Set to not-training mode to disable dropout
	p_encoder.train(False)
	decoder.train(False)
	
	# Run post words through p_encoder
	p_encoder_outputs, p_encoder_hidden = p_encoder(input_batches, input_lengths, None)	

	# Create starting vectors for decoder
	decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
	decoder_hidden = p_encoder_hidden[:decoder.n_layers] + p_encoder_hidden[decoder.n_layers:] # Use last (forward) hidden state from encoder
	
	if USE_CUDA:
		decoder_input = decoder_input.cuda()

	# Store output words and attention states
	decoded_words = []
	
	# Run through decoder
	for di in range(max_length):
		decoder_output, decoder_hidden = decoder(
			decoder_input, decoder_hidden, p_encoder_outputs
		)

		# Choose top word from output
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0].item()
		if ni == EOS_token:
			decoded_words.append('<EOS>')
			break
		else:
			decoded_words.append(q_data.index2word[ni])
			
		# Next input is chosen word
		decoder_input = Variable(torch.LongTensor([ni]))
		if USE_CUDA: decoder_input = decoder_input.cuda()

	# Set back to training mode
	p_encoder.train(True)
	decoder.train(True)

	return decoded_words	

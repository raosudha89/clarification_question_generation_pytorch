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
	decoder_p_attns = torch.zeros(max_length + 1, max_length + 1)
	
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

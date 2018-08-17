import random
from constants import *
from prepare_data import *
from masked_cross_entropy import *
import torch
import torch.nn as nn
from torch.autograd import Variable

def evaluate_randomly(p_data, q_data, triples, encoder, decoder):
	[input_sentence, target_sentence] = random.choice(triples)
	evaluate_and_show_attention(p_data, q_data, encoder, decoder, input_sentence, target_sentence)
	
def evaluate_and_show_attention(p_data, q_data, encoder, decoder, input_sentence, target_sentence=None):
	output_words = evaluate(p_data, q_data, encoder, decoder, input_sentence)
	output_sentence = ' '.join(output_words)
	print('>', input_sentence)
	if target_sentence is not None:
		print('=', target_sentence)
	print('<', output_sentence)

def evaluate_testset(p_data, q_data, encoder, decoder, test_triples, batch_size, out_file):
	input_seqs, target_seqs = preprocess_data(p_data, q_data, test_triples, shuffle=False)
	print_loss_total = 0
	for input_seqs_batch, target_seqs_batch in iterate_minibatches(input_seqs, target_seqs, batch_size):
		input_batches, input_lengths, target_batches, target_lengths = add_padding(input_seqs_batch, target_seqs_batch, USE_CUDA)
		if USE_CUDA:
			input_batches = input_batches.cuda()

		# Run post words through encoder
		encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

		max_target_length = 50
		# Create starting vectors for decoder
		decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size), volatile=True)	
		decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

		if USE_CUDA:
			decoder_input = decoder_input.cuda()
			all_decoder_outputs = all_decoder_outputs.cuda()

		# Run through decoder one time step at a time
		for t in range(max_target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
			all_decoder_outputs[t] = decoder_output

			# Choose top word from output
			topv, topi = decoder_output.data.topk(1)
			decoder_input = topi.squeeze(1) 
		
		for b in range(batch_size):
			#input_words = []
			#output_words = []
			decoded_words = []
			#for t in range(input_lengths[b]):
			#	input_words.append(p_data.index2word[input_batches[t][b].item()])
			#for t in range(target_lengths[b]):
			#	ni = target_batches[t][b].item()
			#	if ni == EOS_token:
			#		output_words.append('<EOS>')
			#		break
			#	else:
			#		output_words.append(q_data.index2word[ni])
			for t in range(max_target_length):
				topv, topi = all_decoder_outputs[t][b].data.topk(1)
				ni = topi[0].item()
				if ni == EOS_token:
					decoded_words.append('<EOS>')
					break
				else:
					decoded_words.append(q_data.index2word[ni])
			#print '> ' + ' '.join(input_words)
			#print '= ' + ' '.join(output_words)
			#print '< ' + ' '.join(decoded_words)
			if out_file:
				out_file.write(' '.join(decoded_words)+'\n')

			#decoder_input = Variable(torch.LongTensor([ni])) 
			#if USE_CUDA: decoder_input = decoder_input.cuda()

		# Loss calculation
		loss = masked_cross_entropy(
			all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
			target_batches.transpose(0, 1).contiguous(), # -> batch x seq
			target_lengths
			)
		print_loss_total += loss.data[0]
	print 'Test loss %.2f' % print_loss_total

def evaluate(p_data, q_data, encoder, decoder, input_seq, max_length=MAX_POST_LEN):
	input_seqs = [indexes_from_sentence(p_data, input_seq)]
	input_lengths = [len(input_seqs)]
	input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
	
	if USE_CUDA:
		input_batches = input_batches.cuda()
		
	# Set to not-training mode to disable dropout
	encoder.train(False)
	decoder.train(False)
	
	# Run post words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)	

	# Create starting vectors for decoder
	decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
	decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:] # Use last (forward) hidden state from encoder
	
	if USE_CUDA:
		decoder_input = decoder_input.cuda()

	# Store output words and attention states
	decoded_words = []
	decoder_p_attns = torch.zeros(max_length + 1, max_length + 1)
	
	# Run through decoder
	for di in range(max_length):
		decoder_output, decoder_hidden = decoder(
			decoder_input, decoder_hidden, encoder_outputs
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
	encoder.train(True)
	decoder.train(True)

	return decoded_words	

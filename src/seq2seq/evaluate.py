import random
from constants import *
from prepare_data import *
from masked_cross_entropy import *
from helper import *
import torch
import torch.nn as nn
from torch.autograd import Variable

def evaluate(word2index, index2word, encoder, decoder, test_data, BATCH_SIZE, out_file):
	input_seqs, input_lens, output_seqs, output_lens = test_data
	total_loss = 0.
	n_batches = len(input_seqs) / BATCH_SIZE
	
	for input_seqs_batch, input_lens_batch, output_seqs_batch, output_lens_batch in \
                iterate_minibatches(input_seqs, input_lens, output_seqs, output_lens, BATCH_SIZE):
		
		if USE_CUDA:
			input_seqs_batch = Variable(torch.LongTensor(np.array(input_seqs_batch)).cuda()).transpose(0, 1)
			output_seqs_batch = Variable(torch.LongTensor(np.array(output_seqs_batch)).cuda()).transpose(0, 1)
		else:
			input_seqs_batch = Variable(torch.LongTensor(np.array(input_seqs_batch))).transpose(0, 1)
			output_seqs_batch = Variable(torch.LongTensor(np.array(output_seqs_batch))).transpose(0, 1)

		# Run post words through encoder
		encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens_batch, None)

		max_output_length = 50
		# Create starting vectors for decoder
		decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * BATCH_SIZE), volatile=True)	
		decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
		all_decoder_outputs = Variable(torch.zeros(max_output_length, BATCH_SIZE, decoder.output_size))
	
		if USE_CUDA:
			decoder_input = decoder_input.cuda()
			all_decoder_outputs = all_decoder_outputs.cuda()
	
		# Run through decoder one time step at a time
		for t in range(max_output_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
			all_decoder_outputs[t] = decoder_output
			# Choose top word from output
			topv, topi = decoder_output.data.topk(1)
			decoder_input = topi.squeeze(1) 
		for b in range(BATCH_SIZE):
			#input_words = []
			#output_words = []
			decoded_words = []
			#for t in range(input_lens_batch[b]):
			#	input_words.append(index2word[input_seqs_batch[t][b].item()])
			#for t in range(output_lens_batch[b]):
			#	ni = output_seqs_batch[t][b].item()
			#	if ni == word2index[EOS_token]:
			#		output_words.append(EOS_token)
			#		break
			#	else:
			#		output_words.append(index2word[ni])
			for t in range(max_output_length):
				topv, topi = all_decoder_outputs[t][b].data.topk(1)
				ni = topi[0].item()
				if ni == word2index[EOS_token]:
					decoded_words.append(EOS_token)
					break
				else:
					decoded_words.append(index2word[ni])
			#print '> ' + ' '.join(input_words)
			#print '= ' + ' '.join(output_words)
			#print '< ' + ' '.join(decoded_words)
			#print 
			if out_file:
				out_file.write(' '.join(decoded_words)+'\n')
	
		# Loss calculation
		loss = masked_cross_entropy(
			all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
			output_seqs_batch.transpose(0, 1).contiguous(), # -> batch x seq
			output_lens_batch
			)
		total_loss += loss.item()
	print 'Loss: %.2f' % (total_loss/n_batches)

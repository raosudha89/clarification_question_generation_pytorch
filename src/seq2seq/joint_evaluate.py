import random
from constants import *
from prepare_pqa_data import *
from masked_cross_entropy import *
import torch
import torch.nn as nn
from torch.autograd import Variable

def evaluate_testset(p_data, q_data, a_data, q_encoder, q_decoder, a_encoder, a_decoder, \
						test_triples, batch_size, ques_out_file, ans_out_file):

	post_seqs, ques_seqs, ans_seqs = preprocess_data(p_data, q_data, a_data, test_triples, shuffle=True)
	print_loss_total = 0
	for post_seqs_batch, ques_seqs_batch, ans_seqs_batch in \
                iterate_minibatches(post_seqs, ques_seqs, ans_seqs, batch_size):
		post_seqs_batch, post_lens, \
		ques_seqs_batch, ques_lens, \
		ans_seqs_batch, ans_lens = add_padding(post_seqs_batch, ques_seqs_batch, ans_seqs_batch, USE_CUDA)

		if USE_CUDA:
			post_seqs_batch = post_seqs_batch.cuda()
			ques_seqs_batch = ques_seqs_batch.cuda()
			ans_seqs_batch = ans_seqs_batch.cuda()

		q_loss = run_decoder(p_data, q_data, q_encoder, q_decoder, \
							post_seqs_batch, post_lens, ques_seqs_batch, ques_lens, ques_out_file)
		a_loss = run_decoder(q_data, a_data, a_encoder, a_decoder, \
							ques_seqs_batch, ques_lens, ans_seqs_batch, ans_lens, ans_out_file)

		print_loss_total += (q_loss + a_loss)
		break
	print 'Test loss %.2f' % print_loss_total

def run_decoder(input_data, output_data, encoder, decoder, input_batches, input_lens, target_batches, target_lens, out_file):
		# Run post words through encoder
		encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

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
		ct = 0		
		for b in range(batch_size):
			input_words = []
			output_words = []
			decoded_words = []
			for t in range(input_lens[b]):
				input_words.append(input_data.index2word[input_batches[t][b].item()])
			for t in range(target_lens[b]):
				ni = target_batches[t][b].item()
				if ni == EOS_token:
					output_words.append('<EOS>')
					break
				else:
					output_words.append(output_data.index2word[ni])
			for t in range(max_target_length):
				topv, topi = all_decoder_outputs[t][b].data.topk(1)
				ni = topi[0].item()
				if ni == EOS_token:
					decoded_words.append('<EOS>')
					break
				else:
					decoded_words.append(output_data.index2word[ni])
			print '> ' + ' '.join(input_words)
			print '= ' + ' '.join(output_words)
			print '< ' + ' '.join(decoded_words)
			print 
			if out_file:
				out_file.write(' '.join(decoded_words)+'\n')
			ct += 1
			if ct > 10:
				break

		# Loss calculation
		loss = masked_cross_entropy(
			all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
			target_batches.transpose(0, 1).contiguous(), # -> batch x seq
			target_lens
			)
		
		return loss.data[0]



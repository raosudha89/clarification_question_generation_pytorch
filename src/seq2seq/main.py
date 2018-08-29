from attn import *
from attnDecoderRNN import *
from constants import *
from encoderRNN import *
from evaluate import *
from helper import *
from train import *
import torch
import torch.optim as optim
from prepare_data import *

def run_seq2seq(train_data, test_data, word2index, index2word, word_embeddings, \
				out_fname, encoder_params_file, decoder_params_file, max_target_length, n_epochs, batch_size, n_layers):
	# Initialize q models
	print 'Initializing models'
	encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers, dropout=DROPOUT)
	decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers)

	# Initialize optimizers
	encoder_optimizer = optim.Adam([par for par in encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
	decoder_optimizer = optim.Adam([par for par in decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

	# Move models to GPU
	if USE_CUDA:
		encoder.cuda()
		decoder.cuda()

	# Keep track of time elapsed and running averages
	start = time.time()
	print_loss_total = 0 # Reset every print_every
	epoch = 0.0

	input_seqs, input_lens, output_seqs, output_lens = train_data
	
	n_batches = len(input_seqs) / batch_size
	teacher_forcing_ratio = 1.0
	#decr = teacher_forcing_ratio/n_epochs
	while epoch < n_epochs:
		epoch += 1
		for input_seqs_batch, input_lens_batch, \
			output_seqs_batch, output_lens_batch in \
                iterate_minibatches(input_seqs, input_lens, output_seqs, output_lens, batch_size):

			start_time = time.time()
			# Run the train function
			loss = train(
				input_seqs_batch, input_lens_batch,
				output_seqs_batch, output_lens_batch,
				encoder, decoder,
				encoder_optimizer, decoder_optimizer, 
				word2index[SOS_token], max_target_length, 
				batch_size, teacher_forcing_ratio 
			)
	
			# Keep track of loss
			print_loss_total += loss
	
		#teacher_forcing_ratio = teacher_forcing_ratio - decr	
		print_loss_avg = print_loss_total / n_batches
		print_loss_total = 0
		print_summary = '%s %d %.4f' % (time_since(start, epoch / n_epochs), epoch, print_loss_avg)
		print(print_summary)
		print 'Epoch: %d' % epoch
		if epoch%10 == 0:
			print 'Saving model params'
			torch.save(encoder.state_dict(), encoder_params_file+'.epoch%d' % epoch)
			torch.save(decoder.state_dict(), decoder_params_file+'.epoch%d' % epoch)
		if (epoch > n_epochs - 10) or (epoch%10 == 0):
			out_file = open(out_fname+'.epoch%d' % int(epoch), 'w')	
			evaluate(word2index, index2word, encoder, decoder, test_data, batch_size, out_file)



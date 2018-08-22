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

def run_seq2seq(train_data, test_data, word2index, index2word, word_embeddings, out_fname, encoder_params_file, decoder_params_file):
	# Initialize q models
	encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, N_LAYERS, dropout=DROPOUT)
	decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, N_LAYERS)

	#if os.path.isfile(encoder_params_file):
	#	print 'Loading saved params...'
	#	encoder.load_state_dict(torch.load(encoder_params_file))
	#	decoder.load_state_dict(torch.load(decoder_params_file))
	#	print 'Done!'

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
	
	n_batches = len(input_seqs) / BATCH_SIZE
	while epoch < N_EPOCHS:
		epoch += 1
		for input_seqs_batch, input_lens_batch, \
			output_seqs_batch, output_lens_batch in \
                iterate_minibatches(input_seqs, input_lens, output_seqs, output_lens, BATCH_SIZE):

			start_time = time.time()
			# Run the train function
			loss = train(
				input_seqs_batch, input_lens_batch,
				output_seqs_batch, output_lens_batch,
				encoder, decoder,
				encoder_optimizer, decoder_optimizer, word2index[SOS_token]
			)
	
			# Keep track of loss
			print_loss_total += loss
		
		print_loss_avg = print_loss_total / n_batches
		print_loss_total = 0
		print_summary = '%s %d %.4f' % (time_since(start, epoch / N_EPOCHS), epoch, print_loss_avg)
		print(print_summary)
		print 'Epoch: %d' % epoch
		if epoch == N_EPOCHS-1:
			print 'Saving model params'
			torch.save(encoder.state_dict(), encoder_params_file)
			torch.save(decoder.state_dict(), decoder_params_file)
		if epoch > N_EPOCHS - 10:
			out_file = open(out_fname+'.epoch%d' % int(epoch), 'w')	
			evaluate(word2index, index2word, encoder, decoder, test_data, BATCH_SIZE, out_file)



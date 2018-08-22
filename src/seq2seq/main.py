from attn import *
from encoderRNN import *
from attnDecoderRNN import *
from helper import *
from train import *
from evaluate import *
from prepare_data import *
from RL_constants import *

def run_seq2seq(train_data, test_data, word2index, index2word, \
			word_embeddings, out_file, encoder_params_file, decoder_params_file):
	# Initialize q models
	encoder = EncoderRNN(len(word2index), hidden_size, word_embeddings, n_layers, dropout=dropout)
	decoder = AttnDecoderRNN(attn_model, hidden_size, len(word2index), word_embeddings, n_layers)

	#if os.path.isfile(encoder_params_file):
	#	print 'Loading saved params...'
	#	encoder.load_state_dict(torch.load(encoder_params_file))
	#	decoder.load_state_dict(torch.load(decoder_params_file))
	#	print 'Done!'

	# Initialize optimizers
	encoder_optimizer = optim.Adam([par for par in encoder.parameters() if par.requires_grad], lr=learning_rate)
	decoder_optimizer = optim.Adam([par for par in decoder.parameters() if par.requires_grad], lr=learning_rate * decoder_learning_ratio)

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
				encoder_optimizer, decoder_optimizer, word2index[SOS_token]
			)
	
			# Keep track of loss
			print_loss_total += loss
		
		print_loss_avg = print_loss_total / n_batches
		print_loss_total = 0
		print_summary = '%s %d %.4f' % (time_since(start, epoch / n_epochs), epoch, print_loss_avg)
		print(print_summary)
		print 'Epoch: %d' % epoch
		if epoch == n_epochs-1:
			print 'Saving model params'
			torch.save(encoder.state_dict(), encoder_params_file)
			torch.save(decoder.state_dict(), decoder_params_file)
		if epoch > n_epochs - 10:
			out_file = open(args.test_pred_question+'.epoch%d' % int(epoch), 'w')	
			evaluate(word2index, index2word, encoder, decoder, test_data, batch_size, out_file)



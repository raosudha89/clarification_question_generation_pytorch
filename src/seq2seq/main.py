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


def run_seq2seq(train_data, test_data, word2index, word_embeddings,
                encoder_params_file, decoder_params_file, max_target_length,
                n_epochs, batch_size, n_layers):
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
    print_loss_total = 0  # Reset every print_every
    epoch = 0.0
    # epoch = 55.0
    # print 'Loading encoded, decoder params'
    # encoder.load_state_dict(torch.load(encoder_params_file+'.epoch%d' % epoch))
    # decoder.load_state_dict(torch.load(decoder_params_file+'.epoch%d' % epoch))

    ids_seqs, input_seqs, input_lens, output_seqs, output_lens = train_data
    
    n_batches = len(input_seqs) / batch_size
    teacher_forcing_ratio = 1.0
    # decr = teacher_forcing_ratio/n_epochs
    prev_test_loss = None
    num_decrease = 0.0
    while epoch < n_epochs:
        epoch += 1
        for ids_seqs_batch, input_seqs_batch, input_lens_batch, \
            output_seqs_batch, output_lens_batch in \
                iterate_minibatches(ids_seqs, input_seqs, input_lens, output_seqs, output_lens, batch_size):

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
    
        # teacher_forcing_ratio = teacher_forcing_ratio - decr
        print_loss_avg = print_loss_total / n_batches
        print_loss_total = 0
        print 'Epoch: %d' % epoch
        print 'Train Set'
        print_summary = '%s %d %.4f' % (time_since(start, epoch / n_epochs), epoch, print_loss_avg)
        print(print_summary)
        print 'Dev Set'
        curr_test_loss = evaluate(word2index, None, encoder, decoder, test_data,
                                  max_target_length, batch_size, None)
        print '%.4f ' % curr_test_loss
        # if prev_test_loss is not None:
        #     diff_test_loss = prev_test_loss - curr_test_loss
        #     if diff_test_loss <= 0:
        #         num_decrease += 1
        #     if num_decrease > 5:
        #         print 'Early stopping'
        #         print 'Saving model params'
        #         torch.save(encoder.state_dict(), encoder_params_file + '.epoch%d' % epoch)
        #         torch.save(decoder.state_dict(), decoder_params_file + '.epoch%d' % epoch)
        #         return
        if epoch % 5 == 0:
            print 'Saving model params'
            torch.save(encoder.state_dict(), encoder_params_file+'.epoch%d' % epoch)
            torch.save(decoder.state_dict(), decoder_params_file+'.epoch%d' % epoch)



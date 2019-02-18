import random
from constants import *
from prepare_data import *
from masked_cross_entropy import *
from helper import *
import torch
import torch.nn as nn
from torch.autograd import Variable


def evaluate_batch(input_batches, input_lens, encoder, decoder,
                   word2index, max_out_len, batch_size,
                   target_batches=None, target_lens=None):

    input_batches = Variable(torch.LongTensor(np.array(input_batches))).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Run post words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size))    
    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
    decoder_outputs = Variable(torch.zeros(max_out_len, batch_size))
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_outputs = decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_out_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.squeeze(1)
        decoder_outputs[t] = topi.squeeze(1)

    decoded_seqs = []
    decoded_lens = []
    for b in range(batch_size):
        decoded_seq = []    
        for t in range(max_out_len):
            topi = decoder_outputs[t][b].data
            idx = int(topi.item())
            if idx == word2index[EOS_token]:
                decoded_seq.append(idx)
                break
            else:
                decoded_seq.append(idx)
        # if target_batches is not None and decoded_seq.count(word2index['<unk>']) > 1:
        #     import pdb
        #     pdb.set_trace()
        #     decoded_seq = target_batches[b]
        #     decoded_lens.append(target_lens[b])
        # else:
        decoded_lens.append(len(decoded_seq))
        decoded_seq += [word2index[PAD_token]]*(max_out_len - len(decoded_seq))
        decoded_seqs.append(decoded_seq)

    return decoded_seqs, decoded_lens    


def evaluate_seq2seq(word2index, index2word, encoder, decoder,
                     id_seqs, input_seqs, input_lens, output_seqs, output_lens,
                     batch_size, max_out_len, out_fname):
    total_loss = 0.
    n_batches = len(input_seqs) / batch_size

    encoder.eval()
    decoder.eval()
    has_ids = True
    if out_fname:
        out_file = open(out_fname, 'w')
        if id_seqs[0] is not None:
            out_ids_file = open(out_fname+'.ids', 'w')
        else:
            has_ids = False
    for id_seqs_batch, input_seqs_batch, input_lens_batch, output_seqs_batch, output_lens_batch in \
            iterate_minibatches(id_seqs, input_seqs, input_lens, output_seqs, output_lens, batch_size, shuffle=False):

        if USE_CUDA:
            input_seqs_batch = Variable(torch.LongTensor(np.array(input_seqs_batch)).cuda()).transpose(0, 1)
            output_seqs_batch = Variable(torch.LongTensor(np.array(output_seqs_batch)).cuda()).transpose(0, 1)
        else:
            input_seqs_batch = Variable(torch.LongTensor(np.array(input_seqs_batch))).transpose(0, 1)
            output_seqs_batch = Variable(torch.LongTensor(np.array(output_seqs_batch))).transpose(0, 1)

        # Run post words through encoder
        encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens_batch, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size))    
        decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]
        all_decoder_outputs = Variable(torch.zeros(max_out_len, batch_size, decoder.output_size))
    
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
    
        # Run through decoder one time step at a time
        for t in range(max_out_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze(1) 
        for b in range(batch_size):
            decoded_words = []
            for t in range(max_out_len):
                topv, topi = all_decoder_outputs[t][b].data.topk(1)
                ni = topi[0].item()
                if ni == word2index[EOS_token]:
                    decoded_words.append(EOS_token)
                    break
                else:
                    decoded_words.append(index2word[ni])
            if out_fname:
                out_file.write(' '.join(decoded_words)+'\n')
                if has_ids:
                    out_ids_file.write(id_seqs_batch[b]+'\n')
    
        # Loss calculation
        loss_fn = torch.nn.NLLLoss()
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            output_seqs_batch.transpose(0, 1).contiguous(),  # -> batch x seq
            output_lens_batch, loss_fn, max_out_len
            )
        total_loss += loss.item()
    print 'Loss: %.2f' % (total_loss/n_batches)

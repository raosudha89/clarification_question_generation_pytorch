import random
from constants import *
from prepare_data import *
from masked_cross_entropy import *
from helper import *
import torch
from torch.autograd import Variable
import pdb


def evaluate_beam_batch(input_seqs, input_lens, encoder, decoder,
                        word2index, index2word, max_out_len, batch_size):
    encoder.eval()
    decoder.eval()

    if USE_CUDA:
        input_seqs_batch = Variable(torch.LongTensor(input_seqs).cuda()).transpose(0, 1)
    else:
        input_seqs_batch = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)

    # Run post words through encoder
    encoder_outputs, encoder_hidden = encoder(input_seqs_batch, input_lens, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Run through decoder one time step at a time
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
    decoder_out_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)
    log_probs, indices = decoder_out_probs.data.topk(BEAM_SIZE)
    prev_decoder_hiddens = [decoder_hidden] * BEAM_SIZE
    prev_backtrack_seqs = torch.zeros(batch_size, BEAM_SIZE, 1)
    for k in range(BEAM_SIZE):
        prev_backtrack_seqs[:, k, 0] = indices[:, k]
    all_EOS = False
    for t in range(1, max_out_len):
        beam_vocab_log_probs = None
        beam_vocab_idx = None
        decoder_hiddens = [None] * BEAM_SIZE
        for k in range(BEAM_SIZE):
            decoder_input = indices[:, k]
            decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs)
            decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)
            vocab_log_probs = Variable(torch.zeros(batch_size, decoder.output_size))
            if USE_CUDA:
                vocab_log_probs = vocab_log_probs.cuda()
            # make sure EOS has no children
            all_EOS = True
            for b in range(batch_size):
                if word2index[EOS_token] in prev_backtrack_seqs[b, k, :t]:
                    vocab_log_probs[b] = log_probs[b, k]
                else:
                    all_EOS = False
                    vocab_log_probs[b] = (log_probs[b, k] * pow(6, 0.7) / pow(t, 0.7) +
                                          decoder_out_log_probs[b]) / pow(t + 1, 0.7) / pow(6, 0.7)

            if all_EOS:
                break
            topv, topi = vocab_log_probs.data.topk(BEAM_SIZE)
            if k == 0:
                beam_vocab_log_probs = topv
                beam_vocab_idx = topi
            else:
                beam_vocab_log_probs = torch.cat((beam_vocab_log_probs, topv), dim=1)
                beam_vocab_idx = torch.cat((beam_vocab_idx, topi), dim=1)
        if all_EOS:
            break
        topv, topi = beam_vocab_log_probs.data.topk(BEAM_SIZE)
        log_probs = topv
        indices = Variable(torch.zeros(batch_size, BEAM_SIZE, dtype=torch.long))
        prev_decoder_hiddens = Variable(torch.zeros(BEAM_SIZE, decoder_hiddens[0].shape[0],
                                                    batch_size, decoder_hiddens[0].shape[2]))
        if USE_CUDA:
            indices = indices.cuda()
            prev_decoder_hiddens = prev_decoder_hiddens.cuda()
        backtrack_seqs = torch.zeros(batch_size, BEAM_SIZE, t + 1)
        for b in range(batch_size):
            indices[b] = torch.index_select(beam_vocab_idx[b], 0, topi[b])
            backtrack_seqs[b, :, t] = indices[b]
            for k in range(BEAM_SIZE):
                prev_decoder_hiddens[k, :, b, :] = decoder_hiddens[topi[b][k] / BEAM_SIZE][:, b, :]
                backtrack_seqs[b, k, :t] = prev_backtrack_seqs[b, topi[b][k] / BEAM_SIZE, :t]
        prev_backtrack_seqs = backtrack_seqs

    decoded_seqs = [[None]*batch_size]*BEAM_SIZE
    decoded_lens = [[max_out_len]*batch_size]*BEAM_SIZE
    for b in range(batch_size):
        for k in range(BEAM_SIZE):
            decoded_seq = []
            for t in range(backtrack_seqs.shape[2]):
                idx = int(backtrack_seqs[b, k, t])
                if idx == word2index[EOS_token]:
                    decoded_seq.append(idx)
                    break
                else:
                    decoded_seq.append(idx)
            decoded_lens[k][b] = len(decoded_seq)
            decoded_seq += [word2index[PAD_token]] * (max_out_len - len(decoded_seq))
            decoded_seqs[k][b] = decoded_seq

    return decoded_seqs, decoded_lens


def evaluate_beam(word2index, index2word, encoder, decoder, id_seqs, input_seqs, input_lens, output_seqs, output_lens,
                  batch_size, max_out_len, out_fname):
    total_loss = 0.
    n_batches = len(input_seqs) / batch_size

    encoder.eval()
    decoder.eval()
    has_ids = True
    if out_fname:
        out_files = [None] * BEAM_SIZE
        out_ids_files = [None] * BEAM_SIZE
        for k in range(BEAM_SIZE):
            out_files[k] = open(out_fname+'.beam%d' % k, 'w')
            if id_seqs[0] is not None:
                out_ids_files[k] = open(out_fname+'.beam%d.ids' % k, 'w')
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
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)
        log_probs, indices = decoder_out_log_probs.data.topk(BEAM_SIZE)
        prev_decoder_hiddens = [decoder_hidden]*BEAM_SIZE
        prev_backtrack_seqs = torch.zeros(batch_size, BEAM_SIZE, 1)
        for k in range(BEAM_SIZE):
            prev_backtrack_seqs[:, k, 0] = indices[:, k]
        backtrack_seqs = None
        for t in range(1, max_out_len):
            beam_vocab_log_probs = None
            beam_vocab_idx = None
            decoder_hiddens = [None]*BEAM_SIZE
            for k in range(BEAM_SIZE):
                decoder_input = indices[:, k]
                decoder_output, decoder_hiddens[k] = decoder(decoder_input, prev_decoder_hiddens[k], encoder_outputs)    
                decoder_out_log_probs = torch.nn.functional.log_softmax(decoder_output, dim=1)
                vocab_log_probs = Variable(torch.zeros(batch_size, decoder.output_size))
                if USE_CUDA:
                    vocab_log_probs = vocab_log_probs.cuda()
                # make sure EOS has no children
                for b in range(batch_size):
                    if word2index[EOS_token] in prev_backtrack_seqs[b, k, :t]:
                        # vocab_log_probs[b] = log_probs[b, k] + decoder_out_log_probs[b][word2index[PAD_token]]
                        vocab_log_probs[b] = log_probs[b, k]
                    else:
                        # vocab_log_probs[b] = log_probs[b, k] + decoder_out_log_probs[b]
                        vocab_log_probs[b] = (log_probs[b, k]*t + decoder_out_log_probs[b])/(t+1)
                        # vocab_log_probs[b] = (log_probs[b, k] * pow(5+t, 0.7) / pow(5+1, 0.7) +
                        #                       decoder_out_log_probs[b]) * pow(5+1, 0.7) / pow(5+t+1, 0.7)
                topv, topi = vocab_log_probs.data.topk(decoder.output_size)
                if k == 0:
                    beam_vocab_log_probs = topv
                    beam_vocab_idx = topi
                else:
                    beam_vocab_log_probs = torch.cat((beam_vocab_log_probs, topv), dim=1)
                    beam_vocab_idx = torch.cat((beam_vocab_idx, topi), dim=1)
            topv, topi = beam_vocab_log_probs.data.topk(BEAM_SIZE)
            indices = Variable(torch.zeros(batch_size, BEAM_SIZE, dtype=torch.long)) 
            prev_decoder_hiddens = Variable(torch.zeros(BEAM_SIZE, decoder_hiddens[0].shape[0],
                                                        batch_size, decoder_hiddens[0].shape[2]))
            if USE_CUDA:
                indices = indices.cuda()
                prev_decoder_hiddens = prev_decoder_hiddens.cuda()
            backtrack_seqs = torch.zeros(batch_size, BEAM_SIZE, t+1)
            for b in range(batch_size):
                indices[b] = torch.index_select(beam_vocab_idx[b], 0, topi[b])
                backtrack_seqs[b, :, t] = indices[b]
                for k in range(BEAM_SIZE):
                    prev_decoder_hiddens[k, :, b, :] = decoder_hiddens[topi[b][k]/decoder.output_size][:, b, :]
                    backtrack_seqs[b, k, :t] = prev_backtrack_seqs[b, topi[b][k]/decoder.output_size, :t]
            prev_backtrack_seqs = backtrack_seqs

        if backtrack_seqs is not None:
            for b in range(batch_size):
                for k in range(BEAM_SIZE):
                    decoded_words = []
                    for t in range(backtrack_seqs.shape[2]):
                        idx = int(backtrack_seqs[b, k, t])
                        if idx == word2index[EOS_token]:
                            decoded_words.append(EOS_token)
                            break
                        else:
                            decoded_words.append(index2word[idx])
                    if out_fname:
                        out_files[k].write(' '.join(decoded_words)+'\n')
                        if has_ids:
                            out_ids_files[k].write(id_seqs_batch[b]+'\n')
    
        loss_fn = torch.nn.NLLLoss()
        # Loss calculation
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            output_seqs_batch.transpose(0, 1).contiguous(),  # -> batch x seq
            output_lens_batch, loss_fn, max_out_len
            )
        total_loss += loss.item()

    if out_fname:
        for k in range(BEAM_SIZE):
            out_files[k].close()
    print 'Loss: %.2f' % (total_loss/n_batches)

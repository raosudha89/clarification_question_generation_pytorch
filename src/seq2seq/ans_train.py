from constants import *
from masked_cross_entropy import *
import numpy as np
import random
import torch
from torch.autograd import Variable
from constants import *


def train(input_batches, input_lens, target_batches, target_lens,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          word2index, args, mode='train'):
    if mode == 'train':
        encoder.train()
        decoder.train()

        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    if USE_CUDA:
        input_batches = Variable(torch.LongTensor(np.array(input_batches)).cuda()).transpose(0, 1)
        target_batches = Variable(torch.LongTensor(np.array(target_batches)).cuda()).transpose(0, 1)
    else:
        input_batches = Variable(torch.LongTensor(np.array(input_batches))).transpose(0, 1)
        target_batches = Variable(torch.LongTensor(np.array(target_batches))).transpose(0, 1)

    # Run post words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

    # Prepare input and output variables
    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

    if USE_CUDA:
        decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * args.batch_size).cuda())
        all_decoder_outputs = Variable(torch.zeros(args.max_ans_len, args.batch_size, decoder.output_size).cuda())
        decoder_outputs = Variable(torch.zeros(args.max_ans_len, args.batch_size).cuda())
    else:
        decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * args.batch_size))
        all_decoder_outputs = Variable(torch.zeros(args.max_ans_len, args.batch_size, decoder.output_size))
        decoder_outputs = Variable(torch.zeros(args.max_ans_len, args.batch_size))

    # Run through decoder one time step at a time
    for t in range(args.max_ans_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_decoder_outputs[t] = decoder_output

        # Teacher Forcing
        decoder_input = target_batches[t]  # Next input is current target
        decoder_outputs[t] = target_batches[t]

        # # Greeding
        # topv, topi = decoder_output.data.topk(1)
        # decoder_outputs[t] = topi.squeeze(1)

    decoded_seqs = []
    decoded_lens = []
    for b in range(args.batch_size):
        decoded_seq = []
        for t in range(args.max_ans_len):
            topi = decoder_outputs[t][b].data
            idx = int(topi.item())
            if idx == word2index[EOS_token]:
                decoded_seq.append(idx)
                break
            else:
                decoded_seq.append(idx)
        decoded_lens.append(len(decoded_seq))
        decoded_seq += [word2index[PAD_token]] * (args.max_ans_len - len(decoded_seq))
        decoded_seqs.append(decoded_seq)

    loss_fn = torch.nn.NLLLoss()
    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lens, loss_fn, args.max_ans_len
    )
    if mode == 'train':
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss, decoded_seqs, decoded_lens

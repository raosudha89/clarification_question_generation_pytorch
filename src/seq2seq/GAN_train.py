from constants import *
from masked_cross_entropy import *
import torch
from torch.autograd import Variable
from helper import *
from utility.RL_evaluate import *
from seq2seq.RL_evaluate import *


def get_decoded_seqs(decoded_seqs, word2index, max_len, batch_size):
    decoded_lens = []
    for b in range(batch_size):
        reached_EOS = False
        for t in range(max_len):
            if reached_EOS:
                decoded_seqs[b][t] = word2index[PAD_token]
            else:
                idx = int(decoded_seqs[b][t])
                if idx == word2index[EOS_token]:
                    reached_EOS = True
                    decoded_lens.append(t+1)
        if not reached_EOS:  # EOS not in seq
            decoded_lens.append(max_len)
    decoded_lens = np.array(decoded_lens)
    return decoded_seqs, decoded_lens


def GAN_train(post_seqs, post_lens, ques_seqs, ques_lens, q_encoder, q_decoder,
              q_encoder_optimizer, q_decoder_optimizer,
              a_encoder, a_decoder,
              context_model, question_model, answer_model, utility_model,
              utility_criterion, word2index, args):
    
    # Zero gradients of both optimizers
    q_encoder_optimizer.zero_grad()
    q_decoder_optimizer.zero_grad()

    q_encoder.train()
    q_decoder.train()

    input_batches = Variable(torch.LongTensor(np.array(post_seqs))).transpose(0, 1)
    target_batches = Variable(torch.LongTensor(np.array(ques_seqs))).transpose(0, 1)

    decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * args.batch_size))
    decoder_outputs = Variable(torch.zeros(args.max_ques_len, args.batch_size, q_decoder.output_size))
    decoder_hiddens = Variable(torch.zeros(args.max_ques_len, args.batch_size, q_decoder.hidden_size))
    decoded_seqs = Variable(torch.zeros(args.batch_size, args.max_ques_len, dtype=torch.long))

    if USE_CUDA:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
        decoded_seqs = decoded_seqs.cuda()
        decoder_input = decoder_input.cuda()
        decoder_outputs = decoder_outputs.cuda()
        decoder_hiddens = decoder_hiddens.cuda()

    # Run post words through encoder
    encoder_outputs, encoder_hidden = q_encoder(input_batches, post_lens, None)

    # Prepare input and output variables
    decoder_hidden = encoder_hidden[:q_decoder.n_layers] + encoder_hidden[q_decoder.n_layers:]

    # Run through decoder one time step at a time
    for t in range(args.max_ques_len):
        decoder_output, decoder_hidden = q_decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_outputs[t] = decoder_output
        decoder_hiddens[t] = torch.sum(decoder_hidden, dim=0)

        # Teacher Forcing
        decoder_input = target_batches[t]  # Next input is current target
        for b in range(args.batch_size):
            decoded_seqs[b][t] = decoder_output[b].topk(1)[1][0]

    loss_fn = torch.nn.NLLLoss()

    # Loss calculation and backpropagation
    xe_loss = masked_cross_entropy(
        decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        ques_lens, loss_fn, args.max_ques_len
    )

    decoded_seqs, decoded_lens = get_decoded_seqs(decoded_seqs, word2index, args.max_ques_len, args.batch_size)
    post_tensor_seqs = Variable(torch.LongTensor(np.array(post_seqs))).cuda()
    pq_pred = torch.cat((post_tensor_seqs, decoded_seqs), 1)
    pql_pred = np.full(args.batch_size, args.max_post_len+args.max_ques_len)
    a_pred, al_pred = evaluate_batch(pq_pred, pql_pred, a_encoder, a_decoder,
                                     word2index, args.max_ans_len, args.batch_size)
    import pdb
    pdb.set_trace()
    u_preds = evaluate_utility(context_model, question_model, answer_model, utility_model,
                               post_seqs, post_lens, decoded_seqs, decoded_lens, a_pred, al_pred, args)

    u_true = torch.FloatTensor(torch.ones(args.batch_size))
    if USE_CUDA:
        u_true = u_true.cuda()
    d_loss = utility_criterion(u_preds, u_true)

    # loss = 0.1*xe_loss + d_loss
    loss = d_loss

    loss.backward()

    q_encoder_optimizer.step()
    q_decoder_optimizer.step()

    return loss

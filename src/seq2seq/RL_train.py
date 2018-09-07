from constants import *
import torch
from torch.autograd import Variable
from helper import *
from utility.RL_evaluate import *
from seq2seq.RL_evaluate import *

def get_decoded_seqs(output_seqs, word2index, max_len, batch_size):
    decoded_seqs = []
    decoded_lens = []
    decoded_seq_masks = []
    for b in range(batch_size):
        decoded_seq = []
        decoded_seq_mask = [0]*max_len
        for t in range(max_len):
            idx = int(output_seqs[t][b])
            if idx == word2index[EOS_token]:
                decoded_seq.append(idx)
                break
            else:
                decoded_seq.append(idx)
                decoded_seq_mask[t] = 1
        decoded_lens.append(len(decoded_seq))
        decoded_seq += [word2index[PAD_token]]*(max_len - len(decoded_seq))
        decoded_seqs.append(decoded_seq)
        decoded_seq_masks.append(decoded_seq_mask)

    decoded_lens = np.array(decoded_lens)
    decoded_seqs = np.array(decoded_seqs)
    decoded_seq_masks = np.array(decoded_seq_masks)
    return decoded_seqs, decoded_lens, decoded_seq_masks


def train(post_seqs, post_lens, ques_seqs, ques_lens, ans_seqs, ans_lens,
          q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer,
          a_encoder, a_decoder, baseline_model, baseline_optimizer, baseline_criterion,
          context_model, question_model, answer_model, utility_model,
          word2index, mixer_delta, args):
    
    # Zero gradients of both optimizers
    q_encoder_optimizer.zero_grad()
    q_decoder_optimizer.zero_grad()
    baseline_optimizer.zero_grad()

    q_encoder.train()
    q_decoder.train()
    baseline_model.train()

    input_batches = Variable(torch.LongTensor(np.array(post_seqs))).transpose(0, 1)
    target_batches = Variable(torch.LongTensor(np.array(ques_seqs))).transpose(0, 1)

    decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * args.batch_size))
    xe_decoder_outputs = Variable(torch.zeros(mixer_delta, args.batch_size, q_decoder.output_size))
    rl_decoder_outputs = Variable(torch.zeros((args.max_ques_len - mixer_delta), args.batch_size, q_decoder.output_size))
    decoder_hiddens = Variable(torch.zeros(args.max_ques_len, args.batch_size, q_decoder.hidden_size))
    output_seqs = Variable(torch.zeros(args.max_ques_len, args.batch_size, dtype=torch.long))

    if USE_CUDA:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
        output_seqs = output_seqs.cuda()
        decoder_input = decoder_input.cuda()
        xe_decoder_outputs = xe_decoder_outputs.cuda()
        rl_decoder_outputs = rl_decoder_outputs.cuda()
        decoder_hiddens = decoder_hiddens.cuda()

    # Run post words through encoder
    encoder_outputs, encoder_hidden = q_encoder(input_batches, post_lens, None)

    # Prepare input and output variables
    decoder_hidden = encoder_hidden[:q_decoder.n_layers] + encoder_hidden[q_decoder.n_layers:]

    # Run through decoder one time step at a time
    for t in range(mixer_delta):
        decoder_output, decoder_hidden = q_decoder(decoder_input, decoder_hidden, encoder_outputs)
        xe_decoder_outputs[t] = decoder_output
        decoder_hiddens[t] = torch.sum(decoder_hidden, dim=0)

        # Teacher Forcing
        decoder_input = target_batches[t]  # Next input is current target
        output_seqs[t] = target_batches[t]

    loss_fn = torch.nn.NLLLoss()

    # # Loss calculation and backpropagation
    xe_loss = masked_cross_entropy(
        xe_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches[:mixer_delta].transpose(0, 1).contiguous(),  # -> batch x seq
        ques_lens, loss_fn, mixer_delta
    )
    xe_loss.backward()

    # Prepare input and output variables
    encoder_outputs, encoder_hidden = q_encoder(input_batches, post_lens, None)
    decoder_hidden = encoder_hidden[:q_decoder.n_layers] + encoder_hidden[q_decoder.n_layers:]
    # decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size)).cuda()

    for t in range(args.max_ques_len-mixer_delta):
        decoder_output, decoder_hidden = q_decoder(decoder_input, decoder_hidden, encoder_outputs)
        rl_decoder_outputs[t] = decoder_output
        decoder_hiddens[mixer_delta+t] = torch.sum(decoder_hidden, dim=0)

        # Sampling
        for b in range(args.batch_size):
            token_dist = torch.nn.functional.softmax(decoder_output[b], dim=0)
            idx = token_dist.multinomial(1, replacement=False).view(-1).data[0]
            decoder_input[b] = idx
            output_seqs[mixer_delta+t][b] = idx

    decoded_seqs, decoded_lens, decoded_seq_masks = get_decoded_seqs(output_seqs, word2index,
                                                                     args.max_ques_len, args.batch_size)

    # Calculate reward
    # reward = calculate_bleu(ques_seqs, ques_lens, decoded_seqs, decoded_lens, index2word, args.max_ques_len)
    pq_pred = np.concatenate((post_seqs, decoded_seqs), axis=1)
    pql_pred = np.full(args.batch_size, args.max_post_len+args.max_ques_len)
    a_pred, al_pred = evaluate_batch(pq_pred, pql_pred, ans_seqs, ans_lens, a_encoder, a_decoder,
                                     word2index, args.max_ans_len, args.batch_size)

    u_preds = evaluate_utility(context_model, question_model, answer_model, utility_model,
                               post_seqs, post_lens, decoded_seqs, decoded_lens, a_pred, al_pred, args)

    reward = u_preds

    # Train baseline reward func model
    baseline_input = torch.sum(decoder_hiddens, dim=0)
    baseline_input = torch.FloatTensor(baseline_input.data.cpu().numpy())
    if USE_CUDA:
        baseline_input = baseline_input.cuda()
    b_reward = baseline_model(baseline_input).squeeze(1)
    b_loss = baseline_criterion(b_reward, reward)
    b_loss.backward()
    baseline_optimizer.step()

    log_probs = calculate_log_probs(
        rl_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        output_seqs[mixer_delta:].transpose(0, 1).contiguous(),  # -> batch x seq
        decoded_lens, loss_fn
    )
    rl_loss = torch.sum(log_probs * (reward.data - b_reward.data)) / args.batch_size
    rl_loss.backward()

    q_encoder_optimizer.step()
    q_decoder_optimizer.step()

    return xe_loss, rl_loss, reward, b_reward
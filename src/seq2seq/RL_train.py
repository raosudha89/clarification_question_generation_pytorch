from constants import *
from masked_cross_entropy import *
import torch
from torch.autograd import Variable
from helper import *


def get_decoded_seqs(output_seqs, word2index, max_len, batch_size):
    decoded_seqs = []
    decoded_lens = []
    decoded_seq_masks = []
    for b in range(batch_size):
        decoded_seq = []
        decoded_seq_mask = [0]*max_len
        log_prob = 0.
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


def train(input_seqs, input_lens, target_seqs, target_lens, encoder, decoder, encoder_optimizer, decoder_optimizer,
            baseline_model, baseline_optimizer, baseline_criterion, word2index, index2word, max_len, batch_size, mixer_delta):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    baseline_optimizer.zero_grad()

    encoder.train()
    decoder.train()
    baseline_model.train()

    input_batches = Variable(torch.LongTensor(np.array(input_seqs))).transpose(0, 1)
    target_batches = Variable(torch.LongTensor(np.array(target_seqs))).transpose(0, 1)

    decoder_input = Variable(torch.LongTensor([word2index[SOS_token]] * batch_size))
    decoder_outputs = Variable(torch.zeros(max_len, batch_size, decoder.output_size))
    decoder_hiddens = Variable(torch.zeros(max_len, batch_size, decoder.hidden_size))
    output_seqs = Variable(torch.zeros(max_len, batch_size, dtype=torch.long))

    if USE_CUDA:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()
        output_seqs = output_seqs.cuda()
        decoder_input = decoder_input.cuda()
        decoder_outputs = decoder_outputs.cuda()
        decoder_hiddens = decoder_hiddens.cuda()

    # Run post words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lens, None)

    # Prepare input and output variables
    decoder_hidden = encoder_hidden[:decoder.n_layers] + encoder_hidden[decoder.n_layers:]

    # Run through decoder one time step at a time
    for t in range(max_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_outputs[t] = decoder_output
        decoder_hiddens[t] = torch.sum(decoder_hidden, dim=0)

        if t < mixer_delta:
            # Teacher Forcing
            decoder_input = target_batches[t]  # Next input is current target
            output_seqs[t] = target_batches[t]
        else:
            # Sampling
            for b in range(batch_size):
                token_dist = torch.nn.functional.softmax(decoder_output[b], dim=0)
                idx = token_dist.multinomial(1, replacement=False).view(-1).data[0]
                decoder_input[b] = idx    
                output_seqs[t][b] = idx

    decoded_seqs, decoded_lens, decoded_seq_masks = get_decoded_seqs(output_seqs, word2index, max_len, batch_size)
    reward = calculate_bleu(target_seqs, target_lens, decoded_seqs, decoded_lens, index2word, max_len)
    decoder_hiddens = torch.sum(decoder_hiddens, dim=0)
    baseline_input = torch.FloatTensor(decoder_hiddens.data.cpu().numpy())
    if USE_CUDA:
        baseline_input = baseline_input.cuda()
    b_reward = baseline_model(baseline_input).squeeze(1)
    b_loss = baseline_criterion(b_reward, reward)
    b_loss.backward()
    baseline_optimizer.step()

    loss_fn = torch.nn.NLLLoss()

    # for t in range(mixer_delta, max_len):
    #     decoder_outputs[t].detach()
    # # Loss calculation and backpropagation
    # xe_loss = masked_cross_entropy(
    #     decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
    #     target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
    #     target_lens, loss_fn, mixer_delta
    # )
    # xe_loss.backward()
    xe_loss = 0.

    for t in range(mixer_delta):
        decoder_outputs[t].detach()

    log_probs = calculate_log_probs(
        decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        output_seqs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lens, loss_fn, mixer_delta
    )
    rl_loss = torch.sum(log_probs * (reward.data - b_reward.data)) / batch_size
    rl_loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return xe_loss, rl_loss, reward, b_reward

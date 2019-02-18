import argparse
import sys
import os
import cPickle as p
import string
import time
import datetime
import math
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

import numpy as np

from seq2seq.encoderRNN import *
from seq2seq.attnDecoderRNN import *
from seq2seq.read_data import *
from seq2seq.prepare_data import *
from seq2seq.masked_cross_entropy import *
from seq2seq.RL_train import *
from seq2seq.RL_evaluate import *
from seq2seq.RL_inference import *
from seq2seq.RL_beam_decoder import *
from seq2seq.baselineFF import *
from seq2seq.GAN_train import *
from utility.RNN import *
from utility.FeedForward import *
from utility.RL_evaluate import *
from utility.RL_train import *
from utility.train import *
from RL_helper import *
from constants import *


def initialize_generator(word_embeddings, word2index):
    print 'Defining encoder decoder models'
    q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
    q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

    a_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
    a_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

    baseline_model = BaselineFF(HIDDEN_SIZE)

    if USE_CUDA:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        q_encoder = q_encoder.to(device)
        q_decoder = q_decoder.to(device)
        a_encoder = a_encoder.to(device)
        a_decoder = a_decoder.to(device)
        baseline_model.to(device)

    # Load encoder, decoder params
    print 'Loading encoder, decoder params'
    q_encoder.load_state_dict(torch.load(args.q_encoder_params))
    q_decoder.load_state_dict(torch.load(args.q_decoder_params))
    a_encoder.load_state_dict(torch.load(args.a_encoder_params))
    a_decoder.load_state_dict(torch.load(args.a_decoder_params))
    print 'Done! '

    q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE)
    q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE * DECODER_LEARNING_RATIO)
    a_encoder_optimizer = optim.Adam([par for par in a_encoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE)
    a_decoder_optimizer = optim.Adam([par for par in a_decoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

    baseline_optimizer = optim.Adam(baseline_model.parameters())
    baseline_criterion = torch.nn.MSELoss()

    return q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer, \
        a_encoder, a_decoder, a_encoder_optimizer, a_decoder_optimizer, \
        baseline_model, baseline_optimizer,  baseline_criterion


def initialize_discriminator(word_embeddings):
    context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
    question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
    answer_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
    utility_model = FeedForward(HIDDEN_SIZE * 3 * 2)
    # utility_model = FeedForward(HIDDEN_SIZE * 2 * 2)
    # utility_model = FeedForward(HIDDEN_SIZE * 2)
    utility_criterion = torch.nn.BCELoss()
    if USE_CUDA:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        context_model.to(device)
        question_model.to(device)
        answer_model.to(device)
        utility_model.to(device)

    # Load utility calculator model params
    print 'Loading utility model params'
    context_model.load_state_dict(torch.load(args.context_params))
    question_model.load_state_dict(torch.load(args.question_params))
    answer_model.load_state_dict(torch.load(args.answer_params))
    utility_model.load_state_dict(torch.load(args.utility_params))
    print 'Done'

    u_optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) +
                             list([par for par in question_model.parameters() if par.requires_grad]) +
                             list([par for par in answer_model.parameters() if par.requires_grad]) +
                             list([par for par in utility_model.parameters() if par.requires_grad]))

    return context_model, question_model, answer_model, utility_model, u_optimizer, utility_criterion


def run_generator(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                  tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens,
                  q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer,
                  a_encoder, a_decoder, a_encoder_optimizer, a_decoder_optimizer,
                  baseline_model, baseline_optimizer, baseline_criterion,
                  context_model, question_model, answer_model, utility_model,
                  word2index, index2word, mixer_delta, args):
    epoch = 0.
    total_xe_loss = 0.
    total_rl_loss = 0.
    total_a_loss = 0.
    total_u_pred = 0.
    total_u_b_pred = 0.

    while epoch < args.g_n_epochs:
        epoch += 1
        for post, pl, ques, ql, pq, pql, ans, al in iterate_minibatches(tr_post_seqs, tr_post_lens,
                                                                        tr_ques_seqs, tr_ques_lens,
                                                                        tr_post_ques_seqs, tr_post_ques_lens,
                                                                        tr_ans_seqs, tr_ans_lens, args.batch_size):
            xe_loss, rl_loss, a_loss, \
                reward, b_reward = GAN_train(post, pl, ques, ql, ans, al,
                                             q_encoder, q_decoder,
                                             q_encoder_optimizer, q_decoder_optimizer,
                                             a_encoder, a_decoder,
                                             a_encoder_optimizer, a_decoder_optimizer,
                                             baseline_model, baseline_optimizer, baseline_criterion,
                                             context_model, question_model, answer_model, utility_model,
                                             word2index, index2word, mixer_delta, args)
            total_xe_loss += xe_loss
            if mixer_delta != args.max_ques_len:
                total_u_pred += reward.data.sum() / args.batch_size
                total_u_b_pred += b_reward.data.sum() / args.batch_size
                total_rl_loss += rl_loss
                total_a_loss += a_loss

    total_xe_loss = total_xe_loss / args.g_n_epochs
    total_rl_loss = total_rl_loss / args.g_n_epochs
    total_a_loss = total_a_loss / args.g_n_epochs
    total_u_pred = total_u_pred / args.g_n_epochs
    total_u_b_pred = total_u_b_pred / args.g_n_epochs
    return total_xe_loss, total_rl_loss, total_a_loss, total_u_pred, total_u_b_pred


def run_discriminator(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                      tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens,
                      q_encoder, q_decoder, a_encoder, a_decoder,
                      baseline_model, baseline_criterion, mixer_delta,
                      context_model, question_model, answer_model,
                      utility_model, optimizer, utility_criterion,
                      word2index, index2word, args):
    epoch = 0.
    total_loss = 0
    total_acc = 0
    context_model.train()
    question_model.train()
    answer_model.train()
    utility_model.train()
    q_encoder.eval()
    q_decoder.eval()
    a_encoder.eval()
    a_decoder.eval()

    while epoch < args.a_n_epochs:
        epoch += 1
        data_size = (len(tr_post_seqs)/args.batch_size)*args.batch_size*2
        post_data = [None]*data_size
        post_len_data = [None]*data_size
        ques_data = [None]*data_size
        ques_len_data = [None]*data_size
        ans_data = [None]*data_size
        ans_len_data = [None]*data_size
        labels_data = [None]*data_size
        batch_num = 0
        for post, pl, ques, ql, pq, pql, ans, al in iterate_minibatches(tr_post_seqs, tr_post_lens,
                                                                        tr_ques_seqs, tr_ques_lens,
                                                                        tr_post_ques_seqs, tr_post_ques_lens,
                                                                        tr_ans_seqs, tr_ans_lens, args.batch_size):
            q_pred, ql_pred, a_pred, al_pred, \
                a_pred_true, al_pred_true = GAN_train(post, pl, ques, ql, ans, al,
                                                      q_encoder, q_decoder,
                                                      None, None,
                                                      a_encoder, a_decoder,
                                                      None, None,
                                                      baseline_model, None, baseline_criterion,
                                                      context_model, question_model, answer_model, utility_model,
                                                      word2index, index2word, mixer_delta, args, mode='test')
            # pq_pred = np.concatenate((post, ques), axis=1)
            # pql_pred = np.full(args.batch_size, args.max_post_len + args.max_ques_len)
            # a_pred_true, al_pred_true = evaluate_batch(pq_pred, pql_pred, a_encoder, a_decoder,
            #                                            word2index, args.max_ans_len, args.batch_size,
            #                                            ans, al)
            # if epoch < 2 and batch_num < 5:
            #     # print 'True Q: %s' % ' '.join([index2word[idx] for idx in ques[0]])
            #     # print 'True A: %s' % ' '.join([index2word[idx] for idx in ans[0]])
            #     print 'Fake Q: %s' % ' '.join([index2word[idx] for idx in q_pred[0]])
            #     print 'Fake A: %s' % ' '.join([index2word[idx] for idx in a_pred[0]])
            # print

            post_data[batch_num*2*args.batch_size:
                      batch_num*2*args.batch_size + args.batch_size] = post
            post_data[batch_num*2*args.batch_size + args.batch_size:
                      batch_num*2*args.batch_size + 2 * args.batch_size] = post
            post_len_data[batch_num*2*args.batch_size:
                          batch_num*2*args.batch_size + args.batch_size] = pl
            post_len_data[batch_num*2*args.batch_size + args.batch_size:
                          batch_num*2*args.batch_size + 2 * args.batch_size] = pl
            ques_data[batch_num*2*args.batch_size:
                      batch_num*2*args.batch_size + args.batch_size] = ques
            ques_data[batch_num*2*args.batch_size + args.batch_size:
                      batch_num*2*args.batch_size + 2 * args.batch_size] = q_pred
            ques_len_data[batch_num*2*args.batch_size:
                          batch_num*2*args.batch_size + args.batch_size] = ql
            ques_len_data[batch_num*2*args.batch_size + args.batch_size:
                          batch_num*2*args.batch_size + 2 * args.batch_size] = ql_pred
            # ans_data[batch_num*2*args.batch_size:
            #          batch_num*2*args.batch_size + args.batch_size] = ans
            ans_data[batch_num*2*args.batch_size:
                     batch_num*2*args.batch_size + args.batch_size] = a_pred_true
            ans_data[batch_num*2*args.batch_size + args.batch_size:
                     batch_num*2*args.batch_size + 2 * args.batch_size] = a_pred
            # ans_len_data[batch_num*2*args.batch_size:
            #              batch_num*2*args.batch_size + args.batch_size] = al
            ans_len_data[batch_num*2*args.batch_size:
                         batch_num*2*args.batch_size + args.batch_size] = al_pred_true
            ans_len_data[batch_num*2*args.batch_size + args.batch_size:
                         batch_num*2*args.batch_size + 2 * args.batch_size] = al_pred
            labels_data[batch_num*2*args.batch_size:
                        batch_num*2*args.batch_size + 2*args.batch_size] = \
                np.concatenate((np.ones(len(post)), np.zeros(len(post))), axis=0)
            batch_num += 1

        permutation = np.random.permutation(len(post_data))
        post_data = np.array(post_data)[permutation]
        post_len_data = np.array(post_len_data)[permutation]
        ques_data = np.array(ques_data)[permutation]
        ques_len_data = np.array(ques_len_data)[permutation]
        ans_data = np.array(ans_data)[permutation]
        ans_len_data = np.array(ans_len_data)[permutation]
        labels_data = np.array(labels_data)[permutation]

        train_data = post_data, post_len_data, ques_data, ques_len_data, ans_data, ans_len_data, labels_data
        train_loss, train_acc = train_fn(context_model, question_model, answer_model, utility_model,
                                         train_data, optimizer, utility_criterion, args)
        total_loss += train_loss
        total_acc += train_acc
    total_loss = total_loss / args.a_n_epochs
    total_acc = total_acc / args.a_n_epochs
    return total_loss, total_acc


def run_model(train_data, test_data, word_embeddings, word2index, index2word, args):
    print 'Preprocessing train data..'
    tr_id_seqs, tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,\
        tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens = preprocess_data(train_data, word2index,
                                                                                         args.max_post_len,
                                                                                         args.max_ques_len,
                                                                                         args.max_ans_len)

    print 'Preprocessing test data..'
    te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, \
        te_post_ques_seqs, te_post_ques_lens, te_ans_seqs, te_ans_lens = preprocess_data(test_data, word2index,
                                                                                         args.max_post_len,
                                                                                         args.max_ques_len,
                                                                                         args.max_ans_len)

    q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer, \
        a_encoder, a_decoder, a_encoder_optimizer, a_decoder_optimizer, \
        baseline_model, baseline_optimizer, baseline_criterion = initialize_generator(word_embeddings, word2index)

    context_model, question_model, answer_model, \
        utility_model, u_optimizer, utility_criterion = initialize_discriminator(word_embeddings)

    start = time.time()

    epoch = 0.
    n_batches = len(tr_post_seqs)/args.batch_size
    mixer_delta = args.max_ques_len
    while epoch < args.n_epochs:
        epoch += 1
        if mixer_delta >= 4:
            mixer_delta = mixer_delta - 2
        total_xe_loss, total_rl_loss, total_a_loss, total_u_pred, total_u_b_pred = \
            run_generator(tr_post_seqs, tr_post_lens,
                          tr_ques_seqs, tr_ques_lens,
                          tr_post_ques_seqs, tr_post_ques_lens,
                          tr_ans_seqs, tr_ans_lens,
                          q_encoder, q_decoder,
                          q_encoder_optimizer, q_decoder_optimizer,
                          a_encoder, a_decoder,
                          a_encoder_optimizer, a_decoder_optimizer,
                          baseline_model, baseline_optimizer,
                          baseline_criterion,
                          context_model, question_model,
                          answer_model, utility_model,
                          word2index, index2word, mixer_delta, args)

        print_xe_loss_avg = total_xe_loss / n_batches
        print_rl_loss_avg = total_rl_loss / n_batches
        print_a_loss_avg = total_a_loss / n_batches
        print_u_pred_avg = total_u_pred / n_batches
        print_u_b_pred_avg = total_u_b_pred / n_batches

        print_summary = 'Generator (RL) %s %d XE_loss: %.4f RL_loss: %.4f A_loss: %.4f U_pred: %.4f B_pred: %.4f' % \
                        (time_since(start, epoch / args.n_epochs), epoch,
                         print_xe_loss_avg, print_rl_loss_avg, print_a_loss_avg, print_u_pred_avg, print_u_b_pred_avg)
        print(print_summary)

        if epoch > -1:
            total_u_loss, total_u_acc = run_discriminator(tr_post_seqs, tr_post_lens,
                                                          tr_ques_seqs, tr_ques_lens,
                                                          tr_post_ques_seqs, tr_post_ques_lens,
                                                          tr_ans_seqs, tr_ans_lens,
                                                          q_encoder, q_decoder, a_encoder, a_decoder,
                                                          baseline_model, baseline_criterion, mixer_delta,
                                                          context_model, question_model, answer_model,
                                                          utility_model, u_optimizer, utility_criterion,
                                                          word2index, index2word, args)

            print_u_loss_avg = total_u_loss
            print_u_acc_avg = total_u_acc

            print_summary = 'Discriminator: %s %d U_loss: %.4f U_acc: %.4f ' % \
                            (time_since(start, epoch / args.n_epochs), epoch, print_u_loss_avg, print_u_acc_avg)
            print(print_summary)

        print 'Saving GAN model params'
        torch.save(q_encoder.state_dict(), args.q_encoder_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(q_decoder.state_dict(), args.q_decoder_params + '.' + args.model + '.epoch%d' % epoch)
        # torch.save(a_encoder.state_dict(), args.a_encoder_params + '.' + args.model + '.epoch%d' % epoch)
        # torch.save(a_decoder.state_dict(), args.a_decoder_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(context_model.state_dict(), args.context_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(question_model.state_dict(), args.question_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(answer_model.state_dict(), args.answer_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(utility_model.state_dict(), args.utility_params + '.' + args.model + '.epoch%d' % epoch)

        #print 'Running evaluation...'
        #out_fname = args.test_pred_question + '.' + args.model + '.epoch%d' % int(epoch)
        #evaluate_beam(word2index, index2word, q_encoder, q_decoder,
        #              te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens,
        #              args.batch_size, args.max_ques_len, out_fname)


def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    index2word = reverse_dict(word2index)

    if args.train_ids is not None:
        train_data = read_data(args.train_context, args.train_question, args.train_answer, args.train_ids,
                               args.max_post_len, args.max_ques_len, args.max_ans_len, mode='train')
                               #count=args.batch_size*10)
    else:
        train_data = read_data(args.train_context, args.train_question, args.train_answer, None,
                               args.max_post_len, args.max_ques_len, args.max_ans_len, mode='train')
    if args.tune_ids is not None:
        test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, args.tune_ids,
                              args.max_post_len, args.max_ques_len, args.max_ans_len, mode='test')
                              #count=args.batch_size*5)
    else:
        test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, None,
                              args.max_post_len, args.max_ques_len, args.max_ans_len, mode='test')

    print 'No. of train_data %d' % len(train_data)
    print 'No. of test_data %d' % len(test_data)
    run_model(train_data, test_data, word_embeddings, word2index, index2word, args)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_context", type = str)
    argparser.add_argument("--train_question", type = str)
    argparser.add_argument("--train_answer", type = str)
    argparser.add_argument("--train_ids", type=str)
    argparser.add_argument("--tune_context", type = str)
    argparser.add_argument("--tune_question", type = str)
    argparser.add_argument("--tune_answer", type = str)
    argparser.add_argument("--tune_ids", type=str)
    argparser.add_argument("--test_context", type = str)
    argparser.add_argument("--test_question", type = str)
    argparser.add_argument("--test_answer", type = str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--test_pred_question", type = str)
    argparser.add_argument("--q_encoder_params", type = str)
    argparser.add_argument("--q_decoder_params", type = str)
    argparser.add_argument("--a_encoder_params", type = str)
    argparser.add_argument("--a_decoder_params", type = str)
    argparser.add_argument("--context_params", type = str)
    argparser.add_argument("--question_params", type = str)
    argparser.add_argument("--answer_params", type = str)
    argparser.add_argument("--utility_params", type = str)
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--max_post_len", type = int, default=300)
    argparser.add_argument("--max_ques_len", type = int, default=50)
    argparser.add_argument("--max_ans_len", type = int, default=50)
    argparser.add_argument("--n_epochs", type = int, default=20)
    argparser.add_argument("--g_n_epochs", type=int, default=1)
    argparser.add_argument("--a_n_epochs", type=int, default=1)
    argparser.add_argument("--batch_size", type = int, default=256)
    argparser.add_argument("--model", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

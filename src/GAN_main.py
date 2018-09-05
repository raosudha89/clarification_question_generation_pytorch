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
from utility.RNN import *
from utility.FeedForward import *
from utility.RL_evaluate import *
from utility.RL_train import *
from RL_helper import *
from constants import *


def update_embs(word2index, word_embeddings):
    word_embeddings[word2index[PAD_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
    word_embeddings[word2index[SOS_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
    word_embeddings[word2index[EOP_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
    word_embeddings[word2index[EOS_token]] = nn.init.normal_(torch.empty(1, len(word_embeddings[0])))
    return word_embeddings


def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    #word_embeddings = update_embs(word2index, word_embeddings) --> updating embs gives poor utility results (0.5 acc)
    index2word = reverse_dict(word2index)

    train_data = read_data(args.train_context, args.train_question, args.train_answer, \
                            args.max_post_len, args.max_ques_len, args.max_ans_len, split='train')
    test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, \
                            args.max_post_len, args.max_ques_len, args.max_ans_len, split='test')
    #test_data = read_data(args.test_context, args.test_question, args.test_answer, \
    #                        args.max_post_len, args.max_ques_len, args.max_ans_len, split='test')

    print 'No. of train_data %d' % len(train_data)
    print 'No. of test_data %d' % len(test_data)
    run_model(train_data, test_data, word_embeddings, word2index, index2word, args)


def run_model(train_data, test_data, word_embeddings, word2index, index2word, args):
    print 'Preprocessing train data..'
    tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens = preprocess_data(train_data, word2index, \
                                                                                args.max_post_len, args.max_ques_len, args.max_ans_len)

    print 'Preprocessing test data..'
    te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, \
    te_post_ques_seqs, te_post_ques_lens, te_ans_seqs, te_ans_lens = preprocess_data(test_data, word2index, \
                                                                                args.max_post_len, args.max_ques_len, args.max_ans_len)

    print 'Defining encoder decoder models'
    q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
    q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

    a_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
    a_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

    if USE_CUDA:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        q_encoder = q_encoder.to(device)
        q_decoder = q_decoder.to(device)
        a_encoder = a_encoder.to(device)
        a_decoder = a_decoder.to(device)

    # Load encoder, decoder params
    print 'Loading encoded, decoder params'
    q_encoder.load_state_dict(torch.load(args.q_encoder_params))
    q_decoder.load_state_dict(torch.load(args.q_decoder_params))
    a_encoder.load_state_dict(torch.load(args.a_encoder_params))
    a_decoder.load_state_dict(torch.load(args.a_decoder_params))

    q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad], lr=LEARNING_RATE)
    q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad], lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

    context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
    question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
    answer_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers=1)
    utility_model = FeedForward(HIDDEN_SIZE*3*2)

    baseline_model = BaselineFF(HIDDEN_SIZE)

    if USE_CUDA:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        context_model.to(device)
        question_model.to(device)
        answer_model.to(device)
        utility_model.to(device)    
        baseline_model.to(device)

    # Load utility calculator model params
    print 'Loading utility model params'
    context_model.load_state_dict(torch.load(args.context_params))
    question_model.load_state_dict(torch.load(args.question_params))
    answer_model.load_state_dict(torch.load(args.answer_params))
    utility_model.load_state_dict(torch.load(args.utility_params))

    u_optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) + \
                            list([par for par in question_model.parameters() if par.requires_grad]) + \
                            list([par for par in answer_model.parameters() if par.requires_grad]) + \
                            list([par for par in utility_model.parameters() if par.requires_grad]))

    baseline_optimizer = optim.Adam(baseline_model.parameters())
    baseline_criterion = torch.nn.MSELoss()

    start = time.time()
    #out_file = None
    #out_fname = args.test_pred_question+'.pretrained'
    #evaluate_beam(word2index, index2word, q_encoder, q_decoder, \
    #                te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, args.batch_size, args.max_ques_len, out_fname)

    epoch = 0.
    n_batches = len(tr_post_seqs)/args.batch_size
    mixer_delta = args.max_ques_len-4
    while epoch < args.n_epochs:
        epoch += 1
        if mixer_delta >= 2:
            mixer_delta = mixer_delta - 2    
        g_n_epochs = 1
        total_xe_loss, total_rl_loss, total_loss, total_u_pred, total_u_b_pred = \
                    run_generator(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                                    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens,
                                    q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer,
                                    a_encoder, a_decoder, baseline_model, baseline_optimizer, baseline_criterion,
                                    context_model, question_model, answer_model, utility_model,
                                    word2index, index2word, g_n_epochs, mixer_delta, args)

        print_loss_avg = total_loss / n_batches
        print_xe_loss_avg = total_xe_loss / n_batches
        print_rl_loss_avg = total_rl_loss / n_batches
        print_u_pred_avg = total_u_pred / n_batches
        print_u_b_pred_avg = total_u_b_pred / n_batches

        print_summary = '%s %d Loss: %.4f XE_loss: %.4f RL_loss: %.4f U_pred: %.4f B_pred: %.4f' % \
                        (time_since(start, epoch / args.n_epochs), epoch, print_loss_avg,
                            print_xe_loss_avg, print_rl_loss_avg, print_u_pred_avg, print_u_b_pred_avg)
        print(print_summary)

        a_n_epochs = 1
        total_u_loss, total_u_pos_acc, total_u_neg_acc = run_discriminator(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                            tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens,
                            q_encoder, q_decoder, a_encoder, a_decoder,
                            context_model, question_model, answer_model, utility_model,
                            u_optimizer, word2index, index2word, a_n_epochs, mixer_delta, args)

        print_u_loss_avg = total_u_loss / n_batches
        print_u_pos_acc_avg = total_u_pos_acc / n_batches
        print_u_neg_acc_avg = total_u_neg_acc / n_batches

        print_summary = '%s %d U_loss: %.4f U_pos_acc: %.4f U_neg_acc: %.4f' % \
                        (time_since(start, epoch / args.n_epochs), epoch, print_u_loss_avg, print_u_pos_acc_avg, print_u_neg_acc_avg)
        print(print_summary)

        if epoch > 0:
            print 'Running evaluation...'
            out_fname = args.test_pred_question+'.epoch%d' % int(epoch)
            evaluate_beam(word2index, index2word, q_encoder, q_decoder, \
                            te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, args.batch_size, args.max_ques_len, out_fname)


def run_generator(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
                    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens, \
                    q_encoder, q_decoder, q_encoder_optimizer, q_decoder_optimizer, \
                    a_encoder, a_decoder, baseline_model, baseline_optimizer, baseline_criterion, \
                    context_model, question_model, answer_model, utility_model, \
                    word2index, index2word, n_epochs, mixer_delta, args):
    epoch = 0.
    total_loss = 0.
    total_xe_loss = 0.
    total_rl_loss = 0.
    total_u_pred = 0.
    total_u_b_pred = 0.
    while epoch < n_epochs:    
        epoch += 1
        for post, pl, ques, ql, pq, pql, ans, al in iterate_minibatches(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                                        tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens, args.batch_size):
            xe_loss, q_log_probs, b_reward, q_pred, ql_pred = train(post, pl, ques, ql, q_encoder, q_decoder,
                                        q_encoder_optimizer, q_decoder_optimizer,
                                        baseline_model, baseline_optimizer,
                                        word2index, index2word, args.max_ques_len,
                                        args.batch_size, mixer_delta)
            pq_pred = np.concatenate((post, q_pred), axis=1)
            pql_pred = np.full((args.batch_size), args.max_post_len+args.max_ques_len)
            a_pred, al_pred = evaluate_batch(pq_pred, pql_pred, ans, al, a_encoder, a_decoder,
                            word2index, args.max_ans_len, args.batch_size)
            u_preds = evaluate_utility(context_model, question_model, answer_model, utility_model,
                            post, pl, q_pred, ql_pred, a_pred, al_pred, args)    
            reward = u_preds
            log_probs = q_log_probs
            rl_loss = torch.sum(log_probs * (reward.data)) / args.batch_size
            #reward  = calculate_bleu(ques, ql, q_pred, ql_pred, index2word, args.max_ques_len)
            #b_loss = baseline_criterion(b_reward, reward)
            #b_loss.backward()
            #baseline_optimizer.step()
            #rl_loss = torch.sum(log_probs * (reward.data-b_reward.data)) / args.batch_size

            total_u_pred += reward.data.sum() / args.batch_size
            total_u_b_pred += b_reward.data.sum() / args.batch_size 
            loss = xe_loss + rl_loss
            total_xe_loss += xe_loss
            total_rl_loss += rl_loss
            total_loss += loss
            loss.backward()
            q_encoder_optimizer.step()
            q_decoder_optimizer.step()
    total_xe_loss = total_xe_loss / n_epochs
    total_rl_loss = total_rl_loss / n_epochs
    total_loss = total_loss / n_epochs
    total_u_pred = total_u_pred / n_epochs
    total_u_b_pred = total_u_b_pred / n_epochs
    return total_xe_loss, total_rl_loss, total_loss, total_u_pred, total_u_b_pred 


def run_discriminator(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                    tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens,
                    q_encoder, q_decoder, a_encoder, a_decoder,
                    context_model, question_model, answer_model, utility_model, optimizer,
                    word2index, index2word, n_epochs, mixer_delta, args):
    epoch = 0.
    total_loss = 0
    total_pos_acc = 0
    total_neg_acc = 0
    while epoch < n_epochs:
        epoch += 1
        for post, pl, ques, ql, pq, pql, ans, al in iterate_minibatches(tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens,
                                        tr_post_ques_seqs, tr_post_ques_lens, tr_ans_seqs, tr_ans_lens, args.batch_size):
            q_pred, ql_pred = evaluate_batch(post, pl, ques, ql, q_encoder, q_decoder,
                                            word2index, args.max_ques_len, args.batch_size)
            pq_pred = np.concatenate((post, q_pred), axis=1)
            pql_pred = np.full((args.batch_size), args.max_post_len+args.max_ques_len)
            a_pred, al_pred = evaluate_batch(pq_pred, pql_pred, ans, al, a_encoder, a_decoder,
                            word2index, args.max_ans_len, args.batch_size)
            q_pos_loss, pos_preds, pos_acc = train_utility(context_model, question_model, answer_model, utility_model,
                                                           optimizer, post, pl, ques, ql, ans, al, args, all_pos=True)
            q_neg_loss, neg_preds, neg_acc = train_utility(context_model, question_model, answer_model, utility_model,
                                            optimizer, post, pl, q_pred, ql_pred, a_pred, al_pred, args, all_pos=False)
            loss = q_pos_loss + q_neg_loss
            total_loss += loss
            total_pos_acc += pos_acc
            total_neg_acc += neg_acc
            loss.backward()
            optimizer.step()
    total_loss = total_loss / n_epochs
    total_pos_acc = total_pos_acc / n_epochs
    total_neg_acc = total_neg_acc / n_epochs
    return total_loss, total_pos_acc, total_neg_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_context", type = str)
    argparser.add_argument("--train_question", type = str)
    argparser.add_argument("--train_answer", type = str)
    argparser.add_argument("--tune_context", type = str)
    argparser.add_argument("--tune_question", type = str)
    argparser.add_argument("--tune_answer", type = str)
    argparser.add_argument("--test_context", type = str)
    argparser.add_argument("--test_question", type = str)
    argparser.add_argument("--test_answer", type = str)
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
    argparser.add_argument("--batch_size", type = int, default=128)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

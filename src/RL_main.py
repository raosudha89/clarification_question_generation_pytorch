import argparse
import cPickle as p
import sys

import torch
import torch.nn as nn
from torch import optim

from seq2seq.encoderRNN import *
from seq2seq.attnDecoderRNN import *
from seq2seq.read_data import *
from seq2seq.prepare_data import *
from seq2seq.RL_train import *
from seq2seq.RL_beam_decoder import *
from seq2seq.baselineFF import *
from utility.RNN import *
from utility.FeedForward import *
from utility.RL_train import *
from RL_helper import *
from constants import *


def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    index2word = reverse_dict(word2index)

    train_data = read_data(args.train_context, args.train_question, args.train_answer, args.train_ids,
                           # args.max_post_len, args.max_ques_len, args.max_ans_len, count=args.batch_size*5)
                           args.max_post_len, args.max_ques_len, args.max_ans_len)
    if args.tune_ids is not None:
        test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, args.tune_ids,
                              args.max_post_len, args.max_ques_len, args.max_ans_len)
    else:
        test_data = read_data(args.tune_context, args.tune_question, args.tune_answer, None,
                              # args.max_post_len, args.max_ques_len, args.max_ans_len, count=args.batch_size*2)
                              args.max_post_len, args.max_ques_len, args.max_ans_len)

    print 'No. of train_data %d' % len(train_data)
    print 'No. of test_data %d' % len(test_data)
    run_model(train_data, test_data, word_embeddings, word2index, index2word, args)


def run_model(train_data, test_data, word_embeddings, word2index, index2word, args):
    print 'Preprocessing train data..'
    tr_id_seqs, tr_post_seqs, tr_post_lens, tr_ques_seqs, tr_ques_lens, \
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

    # out_fname = args.test_pred_question+'.pretrained'
    # evaluate_beam(word2index, index2word, q_encoder, q_decoder,
    #               te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens,
    #               args.batch_size, args.max_ques_len, out_fname)

    q_encoder_optimizer = optim.Adam([par for par in q_encoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE)
    q_decoder_optimizer = optim.Adam([par for par in q_decoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE * DECODER_LEARNING_RATIO)
    a_encoder_optimizer = optim.Adam([par for par in a_encoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE)
    a_decoder_optimizer = optim.Adam([par for par in a_decoder.parameters() if par.requires_grad],
                                     lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

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

    for param in context_model.parameters(): param.requires_grad = False
    for param in question_model.parameters(): param.requires_grad = False
    for param in answer_model.parameters(): param.requires_grad = False
    for param in utility_model.parameters(): param.requires_grad = False

    baseline_optimizer = optim.Adam(baseline_model.parameters())
    baseline_criterion = torch.nn.MSELoss()

    epoch = 0.
    start = time.time()
    n_batches = len(tr_post_seqs)/args.batch_size
    mixer_delta = args.max_ques_len
    while epoch < args.n_epochs:
        epoch += 1
        total_loss = 0.
        total_xe_loss = 0.
        total_rl_loss = 0.
        total_u_pred = 0.
        total_u_b_pred = 0.
        if mixer_delta >= 2:
            mixer_delta = mixer_delta - 2
        batch_num = 0
        for post, pl, ques, ql, pq, pql, ans, al in iterate_minibatches(tr_post_seqs, tr_post_lens,
                                                                        tr_ques_seqs, tr_ques_lens,
                                                                        tr_post_ques_seqs, tr_post_ques_lens,
                                                                        tr_ans_seqs, tr_ans_lens, args.batch_size):
            batch_num += 1
            xe_loss, rl_loss, reward, b_reward = train(post, pl, ques, ql, ans, al,
                                                       q_encoder, q_decoder,
                                                       q_encoder_optimizer, q_decoder_optimizer,
                                                       a_encoder, a_decoder,
                                                       a_encoder_optimizer, a_decoder_optimizer,
                                                       baseline_model, baseline_optimizer, baseline_criterion,
                                                       context_model, question_model, answer_model, utility_model,
                                                       word2index, index2word, mixer_delta, args)
            total_u_pred += reward.data.sum() / args.batch_size
            total_u_b_pred += b_reward.data.sum() / args.batch_size
            total_xe_loss += xe_loss
            total_rl_loss += rl_loss

        print_loss_avg = total_loss / n_batches
        print_xe_loss_avg = total_xe_loss / n_batches
        print_rl_loss_avg = total_rl_loss / n_batches
        print_u_pred_avg = total_u_pred / n_batches
        print_u_b_pred_avg = total_u_b_pred / n_batches
        print_summary = '%s %d Loss: %.4f XE_loss: %.4f RL_loss: %.4f U_pred: %.4f B_pred: %.4f' % \
                        (time_since(start, epoch / args.n_epochs), epoch, print_loss_avg, print_xe_loss_avg,
                         print_rl_loss_avg, print_u_pred_avg, print_u_b_pred_avg)
        print(print_summary)

        print 'Saving RL model params'
        torch.save(q_encoder.state_dict(), args.q_encoder_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(q_decoder.state_dict(), args.q_decoder_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(a_encoder.state_dict(), args.a_encoder_params + '.' + args.model + '.epoch%d' % epoch)
        torch.save(a_decoder.state_dict(), args.a_decoder_params + '.' + args.model + '.epoch%d' % epoch)

        # print 'Running evaluation...'
        # out_fname = args.test_pred_question + '.' + args.model + '.epoch%d' % int(epoch)
        # evaluate_beam(word2index, index2word, q_encoder, q_decoder,
        #               te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens,
        #               args.batch_size, args.max_ques_len, out_fname)
        

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
    argparser.add_argument("--batch_size", type = int, default=128)
    argparser.add_argument("--model", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


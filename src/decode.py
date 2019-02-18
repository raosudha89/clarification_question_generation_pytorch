import argparse
import cPickle as p
import sys
import torch

from seq2seq.encoderRNN import *
from seq2seq.attnDecoderRNN import *
from seq2seq.RL_beam_decoder import *
from seq2seq.diverse_beam_decoder import *
from seq2seq.RL_evaluate import *
from RL_helper import *
from constants import *


def main(args):
    print('Enter main')
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    print('Loaded emb of size %d' % len(word_embeddings))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    index2word = reverse_dict(word2index)

    if args.test_ids is not None:
        test_data = read_data(args.test_context, args.test_question, None, args.test_ids,
                              args.max_post_len, args.max_ques_len, args.max_ans_len, mode='test')
    else:
        test_data = read_data(args.test_context, args.test_question, None, None,
                              args.max_post_len, args.max_ques_len, args.max_ans_len, mode='test')

    print 'No. of test_data %d' % len(test_data)
    run_model(test_data, word_embeddings, word2index, index2word, args)


def run_model(test_data, word_embeddings, word2index, index2word, args):
    print 'Preprocessing test data..'
    te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens, \
        te_post_ques_seqs, te_post_ques_lens, te_ans_seqs, te_ans_lens = preprocess_data(test_data, word2index,
                                                                                         args.max_post_len,
                                                                                         args.max_ques_len,
                                                                                         args.max_ans_len)

    print 'Defining encoder decoder models'
    q_encoder = EncoderRNN(HIDDEN_SIZE, word_embeddings, n_layers=2, dropout=DROPOUT)
    q_decoder = AttnDecoderRNN(HIDDEN_SIZE, len(word2index), word_embeddings, n_layers=2)

    if USE_CUDA:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    q_encoder = q_encoder.to(device)
    q_decoder = q_decoder.to(device)

    # Load encoder, decoder params
    print 'Loading encoded, decoder params'
    if USE_CUDA:
        q_encoder.load_state_dict(torch.load(args.q_encoder_params))
        q_decoder.load_state_dict(torch.load(args.q_decoder_params))
    else:
        q_encoder.load_state_dict(torch.load(args.q_encoder_params, map_location='cpu'))
        q_decoder.load_state_dict(torch.load(args.q_decoder_params, map_location='cpu'))

    out_fname = args.test_pred_question+'.'+args.model
    # out_fname = None
    if args.greedy:
        evaluate_seq2seq(word2index, index2word, q_encoder, q_decoder,
                         te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens,
                         args.batch_size, args.max_ques_len, out_fname)
    elif args.beam:
        evaluate_beam(word2index, index2word, q_encoder, q_decoder,
                      te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens,
                      args.batch_size, args.max_ques_len, out_fname)
    elif args.diverse_beam:
        evaluate_diverse_beam(word2index, index2word, q_encoder, q_decoder,
                              te_id_seqs, te_post_seqs, te_post_lens, te_ques_seqs, te_ques_lens,
                              args.batch_size, args.max_ques_len, out_fname)
    else:
        print 'Please specify mode of decoding: --greedy OR --beam OR --diverse_beam'


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_context", type = str)
    argparser.add_argument("--test_question", type = str)
    argparser.add_argument("--test_answer", type = str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--test_pred_question", type = str)
    argparser.add_argument("--q_encoder_params", type = str)
    argparser.add_argument("--q_decoder_params", type = str)
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--max_post_len", type = int, default=300)
    argparser.add_argument("--max_ques_len", type = int, default=50)
    argparser.add_argument("--max_ans_len", type = int, default=50)
    argparser.add_argument("--n_epochs", type = int, default=20)
    argparser.add_argument("--batch_size", type = int, default=128)
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--greedy", type=bool, default=False)
    argparser.add_argument("--beam", type=bool, default=False)
    argparser.add_argument("--diverse_beam", type=bool, default=False)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

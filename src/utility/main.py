from constants import *
import sys
sys.path.append('src/utility')
from FeedForward import *
from evaluate_utility import *
import random
from RNN import *
import time
import torch
from torch import optim
import torch.nn as nn
import torch.autograd as autograd
from train_utility import *


def update_neg_data(data, index2word):
    ids_seqs, post_seqs, post_lens, ques_seqs, ques_lens, ans_seqs, ans_lens = data
    N = 2
    labels = [0]*int(N*len(post_seqs))
    new_post_seqs = [None]*int(N*len(post_seqs))
    new_post_lens = [None]*int(N*len(post_lens))
    new_ques_seqs = [None]*int(N*len(ques_seqs))
    new_ques_lens = [None]*int(N*len(ques_lens))
    new_ans_seqs = [None]*int(N*len(ans_seqs))
    new_ans_lens = [None]*int(N*len(ans_lens))
    for i in range(len(post_seqs)):
        new_post_seqs[N*i] = post_seqs[i]
        new_post_lens[N*i] = post_lens[i]
        new_ques_seqs[N*i] = ques_seqs[i]
        new_ques_lens[N*i] = ques_lens[i]
        new_ans_seqs[N*i] = ans_seqs[i]
        new_ans_lens[N*i] = ans_lens[i]
        labels[N*i] = 1
        for j in range(1, N):
            r = random.randint(0, len(post_seqs)-1)
            new_post_seqs[N*i+j] = post_seqs[i] 
            new_post_lens[N*i+j] = post_lens[i] 
            new_ques_seqs[N*i+j] = ques_seqs[r] 
            new_ques_lens[N*i+j] = ques_lens[r] 
            new_ans_seqs[N*i+j] = ans_seqs[r] 
            new_ans_lens[N*i+j] = ans_lens[r] 
            labels[N*i+j] = 0

    data = new_post_seqs, new_post_lens, \
                new_ques_seqs, new_ques_lens, \
                new_ans_seqs, new_ans_lens, labels

    return data


def run_utility(train_data, test_data, word_embeddings, index2word, args, n_layers):
    context_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers)
    question_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers)
    answer_model = RNN(len(word_embeddings), len(word_embeddings[0]), n_layers)
    utility_model = FeedForward(HIDDEN_SIZE*3*2)

    if USE_CUDA:
        word_embeddings = autograd.Variable(torch.FloatTensor(word_embeddings).cuda())
    else:
        word_embeddings = autograd.Variable(torch.FloatTensor(word_embeddings))

    context_model.embedding.weight.data.copy_(word_embeddings)
    question_model.embedding.weight.data.copy_(word_embeddings)
    answer_model.embedding.weight.data.copy_(word_embeddings)

    # Fix word embeddings
    context_model.embedding.weight.requires_grad = False
    question_model.embedding.weight.requires_grad = False
    answer_model.embedding.weight.requires_grad = False

    optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) +
                            list([par for par in question_model.parameters() if par.requires_grad]) +
                            list([par for par in answer_model.parameters() if par.requires_grad]) +
                            list([par for par in utility_model.parameters() if par.requires_grad]))

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    if USE_CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        context_model = context_model.to(device)
        question_model = question_model.to(device)
        answer_model = answer_model.to(device)
        utility_model = utility_model.to(device)
        criterion = criterion.to(device)

    train_data = update_neg_data(train_data, index2word)
    test_data = update_neg_data(test_data, index2word)

    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_loss, train_acc = train_fn(context_model, question_model, answer_model, utility_model,
                                         train_data, optimizer, criterion, args)
        valid_loss, valid_acc = evaluate(context_model, question_model, answer_model, utility_model,
                                         test_data, criterion, args)
        print('Epoch %d: Train Loss: %.3f, Train Acc: %.3f, Val Loss: %.3f, Val Acc: %.3f' % \
                (epoch, train_loss, train_acc, valid_loss, valid_acc)) 
        print('Time taken: ', time.time()-start_time)
        # if epoch % 5 == 0:
        print('Saving model params')
        torch.save(context_model.state_dict(), args.context_params+'.epoch%d' % epoch)
        torch.save(question_model.state_dict(), args.question_params+'.epoch%d' % epoch)
        torch.save(answer_model.state_dict(), args.answer_params+'.epoch%d' % epoch)
        torch.save(utility_model.state_dict(), args.utility_params+'.epoch%d' % epoch)


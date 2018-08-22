from FeedForward import *
from RNN import *
from constants import *
from evaluate import *
from train import *
import random
import torch
from torch import optim
import torch.nn as nn
import torch.autograd as autograd

def update_neg_data(train_data):
	post_seqs, post_lens, ques_seqs, ques_lens, ans_seqs, ans_lens = train_data
	labels = [0]*(2*len(post_seqs))
	new_post_seqs = [None]*(2*len(post_seqs))
	new_ques_seqs = [None]*(2*len(ques_seqs))
	new_ans_seqs = [None]*(2*len(ans_seqs))
	for i in range(len(post_seqs)):
		new_post_seqs[2*i] = post_seqs[i]
		new_ques_seqs[2*i] = ques_seqs[i]
		new_ans_seqs[2*i] = ans_seqs[i]
		labels[2*i] = 1
		r = random.randint(0, len(post_seqs)-1)
		new_post_seqs[2*i+1] = post_seqs[i] 
		new_ques_seqs[2*i+1] = ques_seqs[r] 
		new_ans_seqs[2*i+1] = ans_seqs[r] 
		labels[2*i+1] = 0

	train_data = new_post_seqs, new_ques_seqs, new_ans_seqs, labels
	return train_data

def run_utility(train_data, test_data, word_embeddings, context_params, question_params, answer_params, utility_params):
	context_model = RNN(len(word_embeddings), len(word_embeddings[0]))
	question_model = RNN(len(word_embeddings), len(word_embeddings[0]))
	answer_model = RNN(len(word_embeddings), len(word_embeddings[0]))
	utility_model = FeedForward(HIDDEN_SIZE*3)

	word_embeddings = autograd.Variable(torch.FloatTensor(word_embeddings).cuda())
	context_model.embedding.weight.data.copy_(word_embeddings)
	question_model.embedding.weight.data.copy_(word_embeddings)
	answer_model.embedding.weight.data.copy_(word_embeddings)

	# Fix word embeddings
	context_model.embedding.weight.requires_grad = False
	question_model.embedding.weight.requires_grad = False
	answer_model.embedding.weight.requires_grad = False

	optimizer = optim.Adam(list([par for par in context_model.parameters() if par.requires_grad]) + \
							list([par for par in question_model.parameters() if par.requires_grad]) + \
							list([par for par in answer_model.parameters() if par.requires_grad]) + \
							list([par for par in utility_model.parameters() if par.requires_grad]))

	criterion = nn.BCEWithLogitsLoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	context_model = context_model.to(device)
	question_model = question_model.to(device)
	answer_model = answer_model.to(device)
	utility_model = utility_model.to(device)
	criterion = criterion.to(device)

	train_data = update_neg_data(train_data)
	test_data = update_neg_data(test_data)

	for epoch in range(U_N_EPOCHS):
		train_loss, train_acc = train_fn(context_model, question_model, answer_model, utility_model, \
																train_data, optimizer, criterion)
		valid_loss, valid_acc = evaluate(context_model, question_model, answer_model, utility_model, \
                                            					test_data, criterion)
		print 'Epoch %d: Train Loss: %.3f, Train Acc: %.3f, Val Loss: %.3f, Val Acc: %.3f' % \
				(epoch, train_loss, train_acc, valid_loss, valid_acc)   
		if epoch == U_N_EPOCHS-1:
			print 'Saving model params'
			torch.save(context_model.state_dict(), context_params)
			torch.save(question_model.state_dict(), question_params)
			torch.save(answer_model.state_dict(), answer_params)
			torch.save(utility_model.state_dict(), utility_params)


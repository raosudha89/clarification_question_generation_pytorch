# -*- coding: utf-8 -*-
import argparse
import sys
import torch
import torch.autograd as autograd
import random
import torch.utils.data as Data
import cPickle as p

def load_data(contexts_fname, answers_fname):
	contexts_file = open(contexts_fname, 'r')
	answers_file = open(answers_fname, 'r')
	contexts = []
	questions = []
	answers = []
	for line in contexts_file.readlines():
		context, question = line.strip('\n').split('<EOP>')
		contexts.append(context)
		questions.append(question)
	answers = [line.strip('\n') for line in answers_file.readlines()]
	data = []
	for i in range(len(contexts)):
		data.append([contexts[i], questions[i], answers[i], 1])
		r = random.randint(0, len(contexts)-1)
		data.append([contexts[i], questions[r], answers[r], 0])
	random.shuffle(data)
	return data

def main(args):
	train_data = load_data(args.train_contexts_fname, args.train_answers_fname)
	dev_data = load_data(args.tune_contexts_fname, args.tune_answers_fname)
	test_data = load_data(args.test_contexts_fname, args.test_answers_fname)

	p.dump(train_data, open(args.train_data, 'wb'))
	p.dump(dev_data, open(args.tune_data, 'wb'))	
	p.dump(test_data, open(args.test_data, 'wb'))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--train_contexts_fname", type = str)
	argparser.add_argument("--train_answers_fname", type = str)
	argparser.add_argument("--tune_contexts_fname", type = str)
	argparser.add_argument("--tune_answers_fname", type = str)
	argparser.add_argument("--test_contexts_fname", type = str)
	argparser.add_argument("--test_answers_fname", type = str)
	argparser.add_argument("--train_data", type = str)
	argparser.add_argument("--tune_data", type = str)
	argparser.add_argument("--test_data", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


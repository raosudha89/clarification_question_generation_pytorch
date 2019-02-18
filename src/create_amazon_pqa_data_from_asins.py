import argparse
import gzip
import nltk
import pdb
import sys, os
import re
from collections import defaultdict


def parse(path):
	g = gzip.open(path, 'r')
	for l in g:
		yield eval(l)


exception_chars = ['|', '/', '\\', '-', '(', ')', '!', ':', ';', '<', '>']


def preprocess(text):
	text = text.replace('|', ' ')
	text = text.replace('/', ' ')
	text = text.replace('\\', ' ')
	text = text.lower()
	#text = re.sub(r'\W+', ' ', text)
	ret_text = ''
	for sent in nltk.sent_tokenize(text):
		ret_text += ' '.join(nltk.word_tokenize(sent)) + ' '
	return ret_text


def main(args):
	products = {}
	for v in parse(args.metadata_fname):
		if 'description' not in v or 'title' not in v:
			continue
		asin = v['asin']
		title = preprocess(v['title'])
		description = preprocess(v['description'])
		product = title + ' . ' + description
		products[asin] = product

	train_asin_file = open(args.train_asin_fname, 'r')
	train_ans_file = open(args.train_ans_fname, 'w')
	tune_asin_file = open(args.tune_asin_fname, 'w')
	tune_context_file = open(args.tune_context_fname, 'w')
	tune_ques_file = open(args.tune_ques_fname, 'w')
	tune_ans_file = open(args.tune_ans_fname, 'w')
	test_asin_file = open(args.test_asin_fname, 'r')
	test_ans_file = open(args.test_ans_fname, 'w')

	train_asins = []
	test_asins = []
	for line in train_asin_file.readlines():
		train_asins.append(line.strip('\n'))
	for line in test_asin_file.readlines():
		test_asins.append(line.strip('\n'))

	asins = []
	contexts = {}
	questions = {}
	answers = {}

	for v in parse(args.qa_data_fname):
		asin = v['asin']
		if asin not in products or 'answer' not in v:
			continue
		question = preprocess(v['question'])
		answer = preprocess(v['answer'])
		if not answer:
			continue
		asins.append(asin)
		contexts[asin] = products[asin]
		questions[asin] = question
		answers[asin] = answer

	for asin in train_asins:
		train_ans_file.write(answers[asin]+'\n')
	for asin in asins:
		if asin in train_asins or asin in test_asins:
			continue
		tune_asin_file.write(asin+'\n')
		tune_context_file.write(contexts[asin]+'\n')
		tune_ques_file.write(questions[asin]+'\n')
		tune_ans_file.write(answers[asin]+'\n')
	for asin in test_asins:
		test_ans_file.write(answers[asin]+'\n')

	train_asin_file.close()
	train_ans_file.close()
	tune_asin_file.close()
	tune_context_file.close()
	tune_ques_file.close()
	tune_ans_file.close()
	test_asin_file.close()
	test_ans_file.close()


if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_fname", type = str)
	argparser.add_argument("--metadata_fname", type = str)
	argparser.add_argument("--train_asin_fname", type = str)
	argparser.add_argument("--train_ans_fname", type = str)
	argparser.add_argument("--tune_asin_fname", type = str)
	argparser.add_argument("--tune_context_fname", type = str)
	argparser.add_argument("--tune_ques_fname", type = str)
	argparser.add_argument("--tune_ans_fname", type = str)
	argparser.add_argument("--test_asin_fname", type = str)
	argparser.add_argument("--test_ans_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


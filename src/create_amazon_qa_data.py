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
	train_src_file = open(args.train_src_fname, 'w')
	train_tgt_file = open(args.train_tgt_fname, 'w')
	tune_src_file = open(args.tune_src_fname, 'w')
	tune_tgt_file = open(args.tune_tgt_fname, 'w')
	test_src_file = open(args.test_src_fname, 'w')
	test_tgt_file = open(args.test_tgt_fname, 'w')
	
	contexts = []
	answers = []
	for v in parse(args.qa_data_fname):
		asin = v['asin']
		if asin not in products or 'answer' not in v:
			continue
		question = preprocess(v['question'])
		answer = preprocess(v['answer'])
		if not answer:
			continue
		context = ' '.join(products[asin].split()[:250]) + ' <EOP> ' + ' '.join(question.split()[:50])
		contexts.append(context)
		answers.append(answer)
	N = len(contexts)
	for i in range(int(N*0.8)):
		train_src_file.write(contexts[i]+'\n')
		train_tgt_file.write(answers[i]+'\n')
	for i in range(int(N*0.8), int(N*0.9)):
		tune_src_file.write(contexts[i]+'\n')
		tune_tgt_file.write(answers[i]+'\n')
	for i in range(int(N*0.9), N):
		test_src_file.write(contexts[i]+'\n')
		test_tgt_file.write(answers[i]+'\n')
	train_src_file.close()
	train_tgt_file.close()
	tune_src_file.close()
	tune_tgt_file.close()
	test_src_file.close()
	test_tgt_file.close()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_fname", type = str)
	argparser.add_argument("--metadata_fname", type = str)
	argparser.add_argument("--train_src_fname", type = str)
	argparser.add_argument("--train_tgt_fname", type = str)
	argparser.add_argument("--tune_src_fname", type = str)
	argparser.add_argument("--tune_tgt_fname", type = str)
	argparser.add_argument("--test_src_fname", type = str)
	argparser.add_argument("--test_tgt_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


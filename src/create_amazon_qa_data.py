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
	contexts_file = open(args.contexts_fname, 'w')
	answers_file = open(args.answers_fname, 'w')
	for v in parse(args.qa_data_fname):
		asin = v['asin']
		if asin not in products or 'answer' not in v:
			continue
		question = preprocess(v['question'])
		answer = preprocess(v['answer'])
		if not answer:
			continue
		context = ' '.join(products[asin].split()[:250]) + ' <EOP> ' + ' '.join(question.split()[:50])
		contexts_file.write(context+'\n')
		answers_file.write(answer+'\n')
	contexts_file.close()
	answers_file.close()	

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_fname", type = str)
	argparser.add_argument("--metadata_fname", type = str)
	argparser.add_argument("--contexts_fname", type = str)
	argparser.add_argument("--answers_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


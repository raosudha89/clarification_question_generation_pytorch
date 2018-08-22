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
	ret_text = ''
	for sent in nltk.sent_tokenize(text):
		ret_text += ' '.join(nltk.word_tokenize(sent)) + ' '
	return ret_text

def main(args):
	output_file = open(args.output_fname, 'w')
	for v in parse(args.metadata_fname):
		if 'description' not in v or 'title' not in v:
			continue
		title = preprocess(v['title'])
		description = preprocess(v['description'])
		product = title + ' . ' + description
		output_file.write(product+'\n')

	for v in parse(args.qa_data_fname):
		if 'answer' not in v:
			continue
		question = preprocess(v['question'])
		answer = preprocess(v['answer'])
		output_file.write(question+'\n')
		output_file.write(answer+'\n')
	output_file.close()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_fname", type = str)
	argparser.add_argument("--metadata_fname", type = str)
	argparser.add_argument("--output_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


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


def get_brand_info(metadata_fname):
    brand_info = {}
    for v in parse(metadata_fname):
        if 'description' not in v or 'title' not in v:
            continue
        asin = v['asin']
        if 'brand' not in v.keys():
            brand_info[asin] = None
        else:
            brand_info[asin] = v['brand']
    return brand_info


def get_sim_prods(sim_prods_filename, brand_info):
    sim_prods_file = open(sim_prods_filename, 'r')
    sim_prods = {}
    for line in sim_prods_file.readlines():
        parts = line.split()
        asin = parts[0]
        sim_prods[asin] = []
        for prod_id in parts[2:]:
            if brand_info[prod_id] and (brand_info[prod_id] != brand_info[asin]):
                sim_prods[asin].append(prod_id)
        if len(sim_prods[asin]) == 0:
            sim_prods[asin] = parts[10:13]
    return sim_prods


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

	train_asin_file = open(args.train_asin_fname, 'w')
	train_context_file = open(args.train_context_fname, 'w')
	train_ques_file = open(args.train_ques_fname, 'w')
	train_ans_file = open(args.train_ans_fname, 'w')
	tune_asin_file = open(args.tune_asin_fname, 'w')
	tune_context_file = open(args.tune_context_fname, 'w')
	tune_ques_file = open(args.tune_ques_fname, 'w')
	tune_ans_file = open(args.tune_ans_fname, 'w')
	test_asin_file = open(args.test_asin_fname, 'w')
	test_context_file = open(args.test_context_fname, 'w')
	test_ques_file = open(args.test_ques_fname, 'w')
	test_ans_file = open(args.test_ans_fname, 'w')

	brand_info = get_brand_info(args.metadata_fname)
	sim_prod = get_sim_prods(args.sim_prod_fname, brand_info)

	asins = []
	contexts = []
	questions = []
	answers = []
	for v in parse(args.qa_data_fname):
		asin = v['asin']
		if asin not in products or 'answer' not in v:
			continue
		question = preprocess(v['question'])
		answer = preprocess(v['answer'])
		if not answer:
			continue
		asins.append(asin)
		contexts.append(products[asin])
		questions.append(question)
		answers.append(answer)
		for k in range(1, 4):
			src_line += quess[sim_prod[prod_id][k]][(j) % len(quess[sim_prod[prod_id][k]])] + ' <EOQ> '

	N = len(contexts)
	for i in range(int(N*0.8)):
		train_asin_file.write(asins[i]+'\n')
		train_context_file.write(contexts[i]+'\n')
		train_ques_file.write(questions[i]+'\n')
		train_ans_file.write(answers[i]+'\n')
	for i in range(int(N*0.8), int(N*0.9)):
		tune_asin_file.write(asins[i]+'\n')
		tune_context_file.write(contexts[i]+'\n')
		tune_ques_file.write(questions[i]+'\n')
		tune_ans_file.write(answers[i]+'\n')
	for i in range(int(N*0.9), N):
		test_asin_file.write(asins[i]+'\n')
		test_context_file.write(contexts[i]+'\n')
		test_ques_file.write(questions[i]+'\n')
		test_ans_file.write(answers[i]+'\n')

	train_asin_file.close()
	train_context_file.close()
	train_ques_file.close()
	train_ans_file.close()
	tune_asin_file.close()
	tune_context_file.close()
	tune_ques_file.close()
	tune_ans_file.close()
	test_asin_file.close()
	test_context_file.close()
	test_ques_file.close()
	test_ans_file.close()


if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_fname", type = str)
	argparser.add_argument("--metadata_fname", type = str)
	argparser.add_argument("--train_asin_fname", type = str)
	argparser.add_argument("--train_context_fname", type = str)
	argparser.add_argument("--train_ques_fname", type = str)
	argparser.add_argument("--train_ans_fname", type = str)
	argparser.add_argument("--tune_asin_fname", type = str)
	argparser.add_argument("--tune_context_fname", type = str)
	argparser.add_argument("--tune_ques_fname", type = str)
	argparser.add_argument("--tune_ans_fname", type = str)
	argparser.add_argument("--test_asin_fname", type = str)
	argparser.add_argument("--test_context_fname", type = str)
	argparser.add_argument("--test_ques_fname", type = str)
	argparser.add_argument("--test_ans_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


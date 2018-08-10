import argparse
import sys, os
from collections import defaultdict
import csv
import math
import pdb
import random
import numpy as np

MAX_POST_LEN=200
MAX_QUES_LEN=40
MAX_ANS_LEN=40
#MIN_TFIDF=30
MIN_TFIDF=2

def write_to_file(ids, args, posts, question_candidates, answer_candidates, split):

	if split == 'train':
		src_file = open(args.train_src_fname, 'w')
		tgt_file = open(args.train_tgt_fname, 'w')
	elif split == 'tune':
		src_file = open(args.tune_src_fname, 'w')
		tgt_file = open(args.tune_tgt_fname, 'w')
	if split == 'test':
		src_file = open(args.test_src_fname, 'w')
		tgt_file = open(args.test_tgt_fname, 'w')
			
	for k, post_id in enumerate(ids):
		src_line = posts[post_id]+' <EOP> '
		src_line += question_candidates[post_id][0]+' <EOQ> '

		src_file.write(src_line+'\n')	
		tgt_file.write(answer_candidates[post_id][0]+'\n')
	
	src_file.close()
	tgt_file.close()

def trim_by_len(s, max_len):
	s = s.lower().strip()
	words = s.split()
	s = ' '.join(words[:max_len])
	return s

def trim_by_tfidf(posts, p_tf, p_idf):
	for post_id in posts:
		post = []
		words = posts[post_id].split()
		for w in words:
			tf = words.count(w)
			#if p_tf[w]*p_idf[w] >= MIN_TFIDF:
			if tf*p_idf[w] >= MIN_TFIDF:
				post.append(w)
			if len(post) >= MAX_POST_LEN:
				break
		posts[post_id] = ' '.join(post)
	return posts

def read_data(args):
	print("Reading lines...")
	posts = {}
	question_candidates = {}
	answer_candidates = {}
	p_tf = defaultdict(int)
	p_idf = defaultdict(int)
	with open(args.post_data_tsvfile, 'rb') as tsvfile:
		post_reader = csv.reader(tsvfile, delimiter='\t')
		N = 0
		for row in post_reader:
			if N == 0:
				N += 1
				continue
			N += 1
			post_id,title,post = row
			post = title + ' ' + post
			post = post.lower().strip()
			for w in post.split():
				p_tf[w] += 1
			for w in set(post.split()):
				p_idf[w] += 1	
			posts[post_id] = post 

	for w in p_idf:
		p_idf[w] = math.log(N*1.0/p_idf[w])

	posts = trim_by_tfidf(posts, p_tf, p_idf)
	q_tf = defaultdict(int)
	q_idf = defaultdict(int)
	N = 0
	with open(args.qa_data_tsvfile, 'rb') as tsvfile:
		qa_reader = csv.reader(tsvfile, delimiter='\t')
		i = 0
		for row in qa_reader:
			if i == 0:
				i += 1
				continue
			post_id,questions = row[0], row[1:11]
			answers = row[11:21]
			questions = [trim_by_len(question, MAX_QUES_LEN) for question in questions]
			question_candidates[post_id] = questions
			answers = [trim_by_len(answer, MAX_ANS_LEN) for answer in answers]
			answer_candidates[post_id] = answers

	train_ids = [train_id.strip('\n') for train_id in open(args.train_ids_file, 'r').readlines()]
	tune_ids = [tune_id.strip('\n') for tune_id in open(args.tune_ids_file, 'r').readlines()]
	test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]
	
	write_to_file(train_ids, args, posts, question_candidates, answer_candidates, 'train')	
	write_to_file(tune_ids, args, posts, question_candidates, answer_candidates, 'tune')	
	write_to_file(test_ids, args, posts, question_candidates, answer_candidates, 'test')	

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_data_tsvfile", type = str)
	argparser.add_argument("--qa_data_tsvfile", type = str)
	argparser.add_argument("--train_ids_file", type = str)
	argparser.add_argument("--train_src_fname", type = str)
	argparser.add_argument("--train_tgt_fname", type = str)
	argparser.add_argument("--tune_ids_file", type = str)
	argparser.add_argument("--tune_src_fname", type = str)
	argparser.add_argument("--tune_tgt_fname", type = str)
	argparser.add_argument("--test_ids_file", type = str)
	argparser.add_argument("--test_src_fname", type = str)
	argparser.add_argument("--test_tgt_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	read_data(args)
	

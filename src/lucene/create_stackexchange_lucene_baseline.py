import argparse
import sys, os
from collections import defaultdict
import csv
import math
import pdb
import random

def read_data(args):
	print("Reading lines...")
	posts = {}
	question_candidates = {}
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
			posts[post_id] = post 

	with open(args.qa_data_tsvfile, 'rb') as tsvfile:
		qa_reader = csv.reader(tsvfile, delimiter='\t')
		i = 0
		for row in qa_reader:
			if i == 0:
				i += 1
				continue
			post_id,questions = row[0], row[2:11] #Ignore the first question since that is the true question
			question_candidates[post_id] = questions

	test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]

	lucene_out_file = open(args.lucene_output_file, 'w')
	for i, test_id in enumerate(test_ids):
		r = random.randint(0,8)
		lucene_out_file.write(question_candidates[test_id][r] + '\n')
	lucene_out_file.close()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_data_tsvfile", type = str)
	argparser.add_argument("--qa_data_tsvfile", type = str)
	argparser.add_argument("--test_ids_file", type = str)
	argparser.add_argument("--lucene_output_file", type = str)
	args = argparser.parse_args()
	print args
	print ""
	read_data(args)
	

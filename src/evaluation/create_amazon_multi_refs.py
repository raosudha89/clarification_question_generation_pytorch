import argparse
import csv
import sys, os, pdb
import nltk
import time
import random
from collections import defaultdict

def create_refs(test_ids, question_candidates, ref_prefix):
	max_ref_count=0
	for ques_id in test_ids:
		asin = ques_id.split('_')[0]
		N = len(question_candidates[asin])
		max_ref_count = max(max_ref_count, N)
	ref_files = [None]*max_ref_count
	for i in range(max_ref_count):
		ref_files[i] = open(ref_prefix+str(i), 'w')
	for ques_id in test_ids:
		asin = ques_id.split('_')[0]
		N = len(question_candidates[asin])
		for i, ques in enumerate(question_candidates[asin]):
			ref_files[i].write(ques+'\n')
		choices = range(N)
		random.shuffle(choices)
		for j in range(N, max_ref_count):
			r = choices[j%N]
			ref_files[j].write(question_candidates[asin][r]+'\n')

def main(args):
	question_candidates = {}
	test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]

	for fname in os.listdir(args.ques_dir):
		with open(os.path.join(args.ques_dir, fname), 'r') as f:
			asin = fname[:-4].split('_')[0]
			if asin not in question_candidates:
				question_candidates[asin] = []
			ques = f.readline().strip('\n')
			question_candidates[asin].append(ques)
	
	create_refs(test_ids, question_candidates, args.ref_prefix)	

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--ques_dir", type = str)
	argparser.add_argument("--test_ids_file", type = str)
	argparser.add_argument("--ref_prefix", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


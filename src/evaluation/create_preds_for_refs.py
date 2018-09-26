import argparse
import csv
import sys, os, pdb
import nltk
import time

def get_annotations(line):
	set_info, post_id, best, valids, confidence = line.split(',')
	annotator_name = set_info.split('_')[0]
	sitename = set_info.split('_')[1]
	best = int(best)
	valids = [int(v) for v in valids.split()]
	confidence = int(confidence)
	return post_id, annotator_name, sitename, best, valids, confidence

def read_human_annotations(human_annotations_filename):
	human_annotations_file = open(human_annotations_filename, 'r')
	annotations = {}
	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		post_id1, annotator_name1, sitename1, best1, valids1, confidence1 = get_annotations(splits[0])
		post_id2, annotator_name2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
		assert(sitename1 == sitename2)
		assert(post_id1 == post_id2)
		post_id = sitename1+'_'+post_id1
		best_union = list(set([best1, best2]))
		valids_inter = list(set(valids1).intersection(set(valids2)))
		annotations[post_id] = list(set(best_union + valids_inter))
	return annotations


def main(args):
	question_candidates = {}
	model_outputs = []

	test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]
	with open(args.qa_data_tsvfile, 'rb') as tsvfile:
		qa_reader = csv.reader(tsvfile, delimiter='\t')
		i = 0
		for row in qa_reader:
			if i == 0:
				i += 1
				continue
			post_id,questions = row[0], row[1:11]
			question_candidates[post_id] = questions

	annotations = read_human_annotations(args.human_annotations)
	model_output_file = open(args.model_output_file, 'r')
	for line in model_output_file.readlines():
		model_outputs.append(line.strip('\n'))

	pred_file = open(args.model_output_file+'.hasrefs', 'w')
	for i, post_id in enumerate(test_ids):
		if post_id not in annotations:
			continue
		pred_file.write(model_outputs[i]+'\n')
		
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_tsvfile", type = str)
	argparser.add_argument("--human_annotations", type = str)
	argparser.add_argument("--model_output_file", type = str)
	argparser.add_argument("--test_ids_file", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


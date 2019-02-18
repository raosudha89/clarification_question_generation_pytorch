import argparse
import csv
from collections import defaultdict
import sys
import numpy as np
import pdb

generic_levels = {'Yes': 1, 'No': 0}

model_dict = {'ref': 0, 'lucene': 1, 'seq2seq.beam': 2,
              'rl.beam': 3,
              'gan.beam': 4}

model_list = ['ref', 'lucene', 'seq2seq.beam',
              'rl.beam',
              'gan.beam']


def get_avg_score(score_dict, ignore_na=False):
    curr_on_topic_score = 0.
    N = 0
    for score, count in score_dict.iteritems():
        curr_on_topic_score += score * count
        if ignore_na:
            if score != 0:
                N += count
        else:
            N += count
    # print N
    return curr_on_topic_score * 1.0 / N


def main(args):
    num_models = len(model_list)
    generic_scores = [None] * num_models
    asins_so_far = [None] * num_models
    for i in range(num_models):
        generic_scores[i] = defaultdict(int)
        asins_so_far[i] = []

    with open(args.aggregate_results) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['_golden'] == 'true' or row['_unit_state'] == 'golden':
                continue
            asin = row['asin']
            question = row['question']
            model_name = row['model_name']
            if model_name not in model_list:
                continue
            if asin not in asins_so_far[model_dict[model_name]]:
                asins_so_far[model_dict[model_name]].append(asin)
            else:
                print '%s duplicate %s' % (model_name, asin)
                continue
            generic_score = generic_levels[row['on_topic']]
            generic_scores[model_dict[model_name]][generic_score] += 1

    for i in range(num_models):
        print model_list[i]
        print len(asins_so_far[i])
        print 'Avg on generic score: %.2f' % get_avg_score(generic_scores[i])
        print 'Generic:', generic_scores[i]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--aggregate_results", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)
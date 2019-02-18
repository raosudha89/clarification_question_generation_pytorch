import argparse
import csv
from collections import defaultdict
import sys
import numpy as np
import pdb

relevance_levels = {'Completely makes sense': 2,
                    'Somewhat makes sense': 1,
                    'Does not make sense': 0}
is_grammatical_levels = {'Grammatical': 2, 'Comprehensible': 1, 'Incomprehensible': 0}
is_specific_levels = {'Specific to this or the same product from a different manufacturer': 3,
                      'Specific to this or some other similar products': 2,
                      'Generic enough to be applicable to many other products of this type': 1,
                      'Generic enough to be applicable to any product under Home and Kitchen': 0,
                      'N/A (Not Applicable)': 0}
asks_new_info_levels = {'Yes': 1, 'No': 0, 'N/A (Not Applicable)': 0}

model_dict = {'ref': 0, 'lucene': 1, 'seq2seq': 2,
              'seq2seq.generic': 3,
              'seq2seq.specific': 4}

model_list = ['ref', 'lucene', 'seq2seq',
              'seq2seq.generic',
              'seq2seq.specific']


def get_avg_score(score_dict, ignore_na=False):
    curr_relevance_score = 0.
    N = 0
    for score, count in score_dict.iteritems():
        curr_relevance_score += score * count
        if ignore_na:
            if score != 0:
                N += count
        else:
            N += count
    # print N
    return curr_relevance_score * 1.0 / N


def main(args):
    num_models = len(model_list)
    relevance_scores = [None] * num_models
    is_grammatical_scores = [None] * num_models
    is_specific_scores = [None] * num_models
    asks_new_info_scores = [None] * num_models
    asins_so_far = [None] * num_models
    for i in range(num_models):
        relevance_scores[i] = defaultdict(int)
        is_grammatical_scores[i] = defaultdict(int)
        is_specific_scores[i] = defaultdict(int)
        asks_new_info_scores[i] = defaultdict(int)
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
                #print '%s duplicate %s' % (model_name, asin)
                continue
            relevance_score = relevance_levels[row['makes_sense']]
            is_grammatical_score = is_grammatical_levels[row['grammatical']]
            specific_score = is_specific_levels[row['is_specific']]
            asks_new_info_score = asks_new_info_levels[row['new_info']]
            relevance_scores[model_dict[model_name]][relevance_score] += 1
            is_grammatical_scores[model_dict[model_name]][is_grammatical_score] += 1
            if relevance_score != 0 and is_grammatical_score != 0:
                is_specific_scores[model_dict[model_name]][specific_score] += 1
                asks_new_info_scores[model_dict[model_name]][asks_new_info_score] += 1

    for i in range(num_models):
        print model_list[i]
        print len(asins_so_far[i])
        print 'Avg on topic score: %.2f' % get_avg_score(relevance_scores[i])
        print 'Avg grammaticality score: %.2f' % get_avg_score(is_grammatical_scores[i])
        print 'Avg specificity score: %.2f' % get_avg_score(is_specific_scores[i])
        print 'Avg new info score: %.2f' % get_avg_score(asks_new_info_scores[i])
        print
        print 'On topic:', relevance_scores[i]
        print 'Is grammatical: ', is_grammatical_scores[i]
        print 'Is specific: ', is_specific_scores[i]
        print 'Asks new info: ', asks_new_info_scores[i]
        print


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--aggregate_results", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

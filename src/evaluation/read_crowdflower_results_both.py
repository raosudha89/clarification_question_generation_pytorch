import argparse
import csv
from collections import defaultdict
import sys
import numpy as np
import pdb

on_topic_levels = {'Yes': 1, 'No': 0}
is_grammatical_levels = {'Grammatical': 2, 'Comprehensible': 1, 'Incomprehensible': 0}
is_specific_levels = {'Specific pretty much only to this product': 4,
                      'Specific to this and other very similar products (or the same product from a different manufacturer)': 3,
                      'Generic enough to be applicable to many other products of this type': 2,
                      'Generic enough to be applicable to any product under Home and Kitchen': 1,
                      'N/A (Not Applicable)': -1}
asks_new_info_levels = {'Completely': 2, 'Somewhat': 1, 'No': 0, 'N/A (Not Applicable)': -1}
useful_levels = {'Useful enough to be included in the product description': 1,
                 'Useful to a large number of potential buyers (or current users)': 1,
                 'Useful to a small number of potential buyers (or current users)': 1,
                 'Useful only to the person asking the question': 0,
                 'N/A (Not Applicable)': -1}

model_dict = {'ref': 0, 'lucene': 1, 'seq2seq.beam': 2,
              'rl.beam': 3,
              'gan.beam': 4}

model_list = ['ref', 'lucene', 'seq2seq.beam',
              'rl.beam',
              'gan.beam']

frequent_words = ['dimensions', 'dimension', 'size', 'measurements', 'measurement',
                  'weight', 'height', 'width', 'diameter', 'density',
                  'bpa', 'difference', 'thread',
                  'china', 'usa']


def get_avg_score(score_dict, ignore_na=True):
    curr_on_topic_score = 0.
    N = 0
    for score, count in score_dict.iteritems():
        curr_on_topic_score += score * count
        if ignore_na:
            if score != -1:
                N += count
        else:
            N += count
    # print N
    return curr_on_topic_score * 1.0 / N


def main(args):
    num_models = len(model_list)
    on_topic_scores = [None] * num_models
    is_grammatical_scores = [None] * num_models
    is_specific_scores = [None] * num_models
    asks_new_info_scores = [None] * num_models
    useful_scores = [None] * num_models
    asins_so_far = [None] * num_models
    for i in range(num_models):
        on_topic_scores[i] = defaultdict(int)
        is_grammatical_scores[i] = defaultdict(int)
        is_specific_scores[i] = defaultdict(int)
        asks_new_info_scores[i] = defaultdict(int)
        useful_scores[i] = defaultdict(int)
        asins_so_far[i] = []

    with open(args.aggregate_results_v1) as csvfile:
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
            on_topic_score = on_topic_levels[row['on_topic']]
            is_grammatical_score = is_grammatical_levels[row['grammatical']]
            specific_score = is_specific_levels[row['is_specific']]
            asks_new_info_score = asks_new_info_levels[row['new_info']]
            useful_score = useful_levels[row['useful_to_another_buyer']]

            on_topic_scores[model_dict[model_name]][on_topic_score] += 1
            is_grammatical_scores[model_dict[model_name]][is_grammatical_score] += 1
            if on_topic_score != 0 and is_grammatical_score != 0:
                is_specific_scores[model_dict[model_name]][specific_score] += 1
                asks_new_info_scores[model_dict[model_name]][asks_new_info_score] += 1
                useful_scores[model_dict[model_name]][useful_score] += 1

    with open(args.aggregate_results_v2) as csvfile:
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
            on_topic_score = on_topic_levels[row['on_topic']]
            is_grammatical_score = is_grammatical_levels[row['grammatical']]
            specific_score = is_specific_levels[row['is_specific']]
            asks_new_info_score = asks_new_info_levels[row['new_info']]
            useful_score = useful_levels[row['useful_to_another_buyer']]

            on_topic_scores[model_dict[model_name]][on_topic_score] += 1
            is_grammatical_scores[model_dict[model_name]][is_grammatical_score] += 1
            if on_topic_score != 0 and is_grammatical_score != 0:
                is_specific_scores[model_dict[model_name]][specific_score] += 1
                asks_new_info_scores[model_dict[model_name]][asks_new_info_score] += 1
                useful_scores[model_dict[model_name]][useful_score] += 1

    for i in range(num_models):
        print model_list[i]
        print len(asins_so_far[i])
        print 'Avg on topic score: %.2f' % get_avg_score(on_topic_scores[i])
        print 'Avg grammaticality score: %.2f' % get_avg_score(is_grammatical_scores[i])
        print 'Avg specificity score: %.2f' % get_avg_score(is_specific_scores[i])
        print 'Avg new info score: %.2f' % get_avg_score(asks_new_info_scores[i])
        print 'Avg useful score: %.2f' % get_avg_score(useful_scores[i])
        print
        print 'On topic:', on_topic_scores[i]
        print 'Is grammatical: ', is_grammatical_scores[i]
        print 'Is specific: ', is_specific_scores[i]
        print 'Asks new info: ', asks_new_info_scores[i]
        print 'Useful: ', useful_scores[i]
        print


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--aggregate_results_v1", type = str)
    argparser.add_argument("--aggregate_results_v2", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

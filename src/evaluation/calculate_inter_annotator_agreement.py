import argparse
import csv
from collections import defaultdict
import sys
import numpy as np
import pdb

on_topic_levels = {'Yes': 1, 'No': 0}
is_grammatical_levels = {'Grammatical': 1, 'Comprehensible': 1, 'Incomprehensible': 0}
is_specific_levels = {'Specific pretty much only to this product': 4,
                      'Specific to this and other very similar products (or the same product from a different manufacturer)': 3,
                      'Generic enough to be applicable to many other products of this type': 2,
                      'Generic enough to be applicable to any product under Home and Kitchen': 1,
                      'N/A (Not Applicable)': -1}
asks_new_info_levels = {'Completely': 1, 'Somewhat': 1, 'No': 0, 'N/A (Not Applicable)': -1}
useful_levels = {'Useful enough to be included in the product description': 4,
                 'Useful to a large number of potential buyers (or current users)': 3,
                 'Useful to a small number of potential buyers (or current users)': 2,
                 'Useful only to the person asking the question': 1,
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
        if score != -1:
            curr_on_topic_score += score * count
        if ignore_na:
            if score != -1:
                N += count
        else:
            N += count
    #print N
    return curr_on_topic_score * 1.0 / N


def main(args):
    num_models = len(model_list)
    on_topic_conf_scores = []
    is_grammatical_conf_scores = []
    is_specific_conf_scores = []
    asks_new_info_conf_scores = []
    useful_conf_scores = []

    asins_so_far = [None] * num_models
    for i in range(num_models):
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
            on_topic_score = on_topic_levels[row['on_topic']]
            on_topic_conf_score = float(row['on_topic:confidence'])
            is_grammatical_score = is_grammatical_levels[row['grammatical']]
            is_grammatical_conf_score = float(row['grammatical:confidence'])
            is_specific_conf_score = float(row['is_specific:confidence'])
            asks_new_info_score = asks_new_info_levels[row['new_info']]
            asks_new_info_conf_score = float(row['new_info:confidence'])
            useful_conf_score = float(row['useful_to_another_buyer:confidence'])

            on_topic_conf_scores.append(on_topic_conf_score)
            is_grammatical_conf_scores.append(is_grammatical_conf_score)
            if on_topic_score != 0 and is_grammatical_score != 0:
                is_specific_conf_scores.append(is_specific_conf_score)
                asks_new_info_conf_scores.append(asks_new_info_conf_score)
                if asks_new_info_score != 0:
                    useful_conf_scores.append(useful_conf_score)

    print 'On topic confidence: %.4f' % (sum(on_topic_conf_scores)/float(len(on_topic_conf_scores)))
    print 'Is grammatical confidence: %.4f' % (sum(is_grammatical_conf_scores)/float(len(is_grammatical_conf_scores)))
    print 'Asks new info confidence: %.4f' % (sum(asks_new_info_conf_scores)/float(len(asks_new_info_conf_scores)))
    print 'Useful confidence: %.4f' % (sum(useful_conf_scores)/float(len(useful_conf_scores)))
    print 'Specificity confidence: %.4f' % (sum(is_specific_conf_scores)/float(len(is_specific_conf_scores)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--aggregate_results", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

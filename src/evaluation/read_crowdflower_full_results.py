import argparse
import csv
from collections import defaultdict
import sys
import numpy as np
import pdb

on_topic_levels = {'Yes': 1, 'No': 0}
is_grammatical_levels = {'Grammatical': 2, 'Comprehensible': 1, 'Incomprehensible': 0}
is_specific_levels = {'Specific pretty much only to this product': 2,
                      'Specific to this and other very similar products (or the same product from a different manufacturer)': 2,
                      'Generic enough to be applicable to many other products of this type': 1,
                      'Generic enough to be applicable to any product under Home and Kitchen': 1,
                      'N/A (Not Applicable)': 0}
asks_new_info_levels = {'Completely': 3, 'Somewhat': 2, 'No': 1, 'N/A (Not Applicable)': 0}
useful_levels = {'Useful enough to be included in the product description': 4,
                 'Useful to a large number of potential buyers (or current users)': 3,
                 'Useful to a small number of potential buyers (or current users)': 2,
                 'Useful only to the person asking the question': 1,
                 'N/A (Not Applicable)': 0}

model_dict = {'ref': 0, 'lucene': 1, 'seq2seq.diverse_beam': 2,
              'RL_selfcritic.epoch8.diverse_beam': 3,
              'GAN_selfcritic_pred_ans_3perid.epoch8.diverse_beam': 4,
              'seq2seq.beam': 5}

model_list = ['ref', 'lucene', 'seq2seq.diverse_beam',
              'RL_selfcritic.epoch8.diverse_beam',
              'GAN_selfcritic_pred_ans_3perid.epoch8.diverse_beam',
              'seq2seq.beam']

frequent_words = ['dimensions', 'dimension', 'size', 'measurements', 'measurement',
                  'weight', 'height', 'width', 'diameter', 'density',
                  'bpa', 'difference', 'thread',
                  'china', 'usa']


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
    print N
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

    with open(args.full_results) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['_golden'] == 'true':
                continue
            question = row['question']
            model_name = row['model_name']
            on_topic_score = on_topic_levels[row['on_topic']]
            is_grammatical_score = is_grammatical_levels[row['grammatical']]
            specific_score = is_specific_levels[row['is_specific']]
            asks_new_info_score = asks_new_info_levels[row['new_info']]
            useful_score = useful_levels[row['useful_to_another_buyer']]
            # for w in frequent_words:
            #     if w in question.split():
            #         specific_score = 2
            if on_topic_score == 0:
                specific_score = 0
                asks_new_info_score = 0
                useful_score = 0
            if is_grammatical_score == 0:
                specific_score = 0
                asks_new_info_score = 0
                useful_score = 0
            on_topic_scores[model_dict[model_name]][on_topic_score] += 1
            is_grammatical_scores[model_dict[model_name]][is_grammatical_score] += 1
            is_specific_scores[model_dict[model_name]][specific_score] += 1
            asks_new_info_scores[model_dict[model_name]][asks_new_info_score] += 1
            useful_scores[model_dict[model_name]][useful_score] += 1
    for i in range(num_models):
        print model_list[i]
        print 'Avg on topic score: %.2f' % get_avg_score(on_topic_scores[i])
        print 'Avg grammaticality score: %.2f' % get_avg_score(is_grammatical_scores[i])
        print 'Avg specificity score: %.2f' % get_avg_score(is_specific_scores[i], ignore_na=True)
        print 'Avg new info score: %.2f' % get_avg_score(asks_new_info_scores[i])
        print 'Avg useful score: %.2f' % get_avg_score(useful_scores[i])
        print
        # print 'On topic:', on_topic_scores[i]
        # print 'Is grammatical: ', is_grammatical_scores[i]
        # print 'Is specific: ', is_specific_scores[i]
        # print 'Asks new info: ', asks_new_info_scores[i]
        # print 'Useful: ', useful_scores[i]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--full_results", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)
import argparse
import csv
from collections import defaultdict
import sys
import pdb
import numpy as np

specificity_levels = ['Question A is more specific',
                      'Question B is more specific',
                      'Both are at the same level of specificity',
                      'N/A: One or both questions are not applicable to the product']


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
    titles = {}
    descriptions = {}
    cand_ques_dict = defaultdict(list)
    cand_scores_dict = defaultdict(list)
    with open(args.aggregate_results) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['_unit_state'] != 'finalized':
                continue
            asin = row['asin']
            titles[asin] = row['title']
            descriptions[asin] = row['description']
            question_a = row['question_a']
            question_b = row['question_b']
            confidence = row['on_topic:confidence']

            if question_a not in cand_ques_dict[asin]:
                cand_ques_dict[asin].append(question_a)
                cand_scores_dict[asin].append([])
            if question_b not in cand_ques_dict[asin]:
                cand_ques_dict[asin].append(question_b)
                cand_scores_dict[asin].append([])

            if row['on_topic'] == 'Question A is more specific':
                cand_scores_dict[asin][cand_ques_dict[asin].index(question_a)].append(float(confidence))
                cand_scores_dict[asin][cand_ques_dict[asin].index(question_b)].append((1 - float(confidence)))
            elif row['on_topic'] == 'Question B is more specific':
                cand_scores_dict[asin][cand_ques_dict[asin].index(question_b)].append(float(confidence))
                cand_scores_dict[asin][cand_ques_dict[asin].index(question_a)].append((1 - float(confidence)))
            elif row['on_topic'] == 'Both are at the same level of specificity':
                cand_scores_dict[asin][cand_ques_dict[asin].index(question_b)].append(0.5)
                cand_scores_dict[asin][cand_ques_dict[asin].index(question_a)].append(0.5)
            else:
                print 'ID: %s has irrelevant question' % asin

    for asin in titles:
        print asin
        print titles[asin]
        print descriptions[asin]
        for i, ques in enumerate(cand_ques_dict[asin]):
            print np.mean(cand_scores_dict[asin][i]), cand_scores_dict[asin][i], ques
        print


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--aggregate_results", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)
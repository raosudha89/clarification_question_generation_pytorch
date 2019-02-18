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
    curr_asin = None
    curr_a = None
    curr_b = None
    a_scores = []
    b_scores = []
    ab_scores = []
    with open(args.full_results) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['_golden'] == 'true' or row['_tainted'] == 'true':
                continue
            asin = row['asin']
            titles[asin] = row['title']
            descriptions[asin] = row['description']
            question_a = row['question_a']
            question_b = row['question_b']
            trust = row['_trust']

            if curr_asin is None:
                curr_asin = asin
                curr_a = question_a
                curr_b = question_b
            elif asin != curr_asin and curr_a != question_a and curr_b != question_b:
                a_score = np.sum(a_scores)/5
                b_score = np.sum(b_scores)/5
                ab_score = np.sum(ab_scores)/5
                cand_scores_dict[curr_asin][cand_ques_dict[curr_asin].index(curr_a)].append(a_score + 0.5*ab_score)
                cand_scores_dict[curr_asin][cand_ques_dict[curr_asin].index(curr_b)].append(b_score + 0.5*ab_score)
                curr_asin = asin
                curr_a = question_a
                curr_b = question_b
                a_scores = []
                b_scores = []
                ab_scores = []

            if question_a not in cand_ques_dict[asin]:
                cand_ques_dict[asin].append(question_a)
                cand_scores_dict[asin].append([])
            if question_b not in cand_ques_dict[asin]:
                cand_ques_dict[asin].append(question_b)
                cand_scores_dict[asin].append([])

            if row['on_topic'] == 'Question A is more specific':
                a_scores.append(float(trust))
            elif row['on_topic'] == 'Question B is more specific':
                b_scores.append(float(trust))
            elif row['on_topic'] == 'Both are at the same level of specificity':
                ab_scores.append(float(trust))
            else:
                print 'ID: %s has irrelevant question' % asin

    corr = 0
    total = 0
    fp, tp, fn, tn = 0, 0, 0, 0
    for asin in titles:
        print asin
        print titles[asin]
        print descriptions[asin]
        for i, ques in enumerate(cand_ques_dict[asin]):
            true_v = np.mean(cand_scores_dict[asin][i])
            pred_v = np.mean(cand_scores_dict[asin][i][:int(len(cand_scores_dict[asin][i])/2)])
            print true_v, cand_scores_dict[asin][i], ques
            print pred_v
            if true_v < 0.5:
                true_l = 0
            else:
                true_l = 1
            if pred_v < 0.5:
                pred_l = 0
            else:
                pred_l = 1
            if true_l == pred_l:
                corr += 1
            if true_l == 0 and pred_l == 1:
                fp += 1
            if true_l == 0 and pred_l == 0:
                tn += 1
            if true_l == 1 and pred_l == 0:
                fn += 1
            if true_l == 1 and pred_l == 1:
                tp += 1
            total += 1
        print
    print 'accuracy'
    print corr*1.0/total
    print tp, fp, fn, tn


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--full_results", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

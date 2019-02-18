import sys
import argparse
from collections import defaultdict


def main(args):
    train_ids_file = open(args.train_ids, 'r')
    train_answer_file = open(args.train_answer, 'r')
    train_candqs_ids_file = open(args.train_candqs_ids, 'r')
    train_answer_candqs_file = open(args.train_answer_candqs, 'w')
    train_ids = []
    uniq_id_ct = defaultdict(int)
    for line in train_ids_file.readlines():
        curr_id = line.strip('\n')
        uniq_id_ct[curr_id] += 1
        train_ids.append(curr_id+'_'+str(uniq_id_ct[curr_id]))
    i = 0
    train_answers = {}
    for line in train_answer_file.readlines():
        train_answers[train_ids[i]] = line.strip('\n')
        i += 1
    for line in train_candqs_ids_file.readlines():
        curr_id = line.strip('\n')
        try:
            train_answer_candqs_file.write(train_answers[curr_id])
        except:
            import pdb
            pdb.set_trace()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train_ids", type=str)
    argparser.add_argument("--train_answer", type=str)
    argparser.add_argument("--train_candqs_ids", type=str)
    argparser.add_argument("--train_answer_candqs", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

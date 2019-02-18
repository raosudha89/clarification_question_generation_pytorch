import sys
import argparse
from collections import defaultdict


def main(args):
    test_output_file = open(args.test_output, 'r')
    test_ids_file = open(args.test_ids, 'r')
    test_output_perid_file = open(args.test_output_perid, 'w')

    # ongoing_id = None
    id_seq_in_output = []
    data_ct_per_id = defaultdict(int)
    for line in test_ids_file.readlines():
        curr_id = line.strip('\n')
        data_ct_per_id[curr_id] += 1
        if args.max_per_id is not None:
            if data_ct_per_id[curr_id] <= args.max_per_id:
                id_seq_in_output.append(curr_id)
        else:
            id_seq_in_output.append(curr_id)
    i = 0
    ongoing_id = None
    test_output_lines = test_output_file.readlines()
    test_output_perid = []
    for line in test_output_lines:
        if id_seq_in_output[i] != ongoing_id:
            test_output_perid.append(line)
            ongoing_id = id_seq_in_output[i]
        i += 1
    total_count = (len(test_output_perid) / args.batch_size - 1) * args.batch_size
    print total_count
    print len(test_output_perid)
    for line in test_output_perid[:total_count]:
        test_output_perid_file.write(line)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_output", type=str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--test_output_perid", type=str)
    argparser.add_argument("--max_per_id", type=int, default=None)
    argparser.add_argument("--batch_size", type=int)
    args = argparser.parse_args()
    print args
    print ""
    main(args)
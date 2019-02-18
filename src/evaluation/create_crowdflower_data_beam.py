import argparse
import gzip
import nltk
import pdb
import sys
from collections import defaultdict
import csv
import random


model_list = ['ref', 'lucene', 'seq2seq.beam',
              'rl.beam',
              'gan.beam']
model_dict = {'ref': 0, 'lucene': 1, 'seq2seq.beam': 2,
              'rl.beam': 3,
              'gan.beam': 4}


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def main(args):
    csv_file = open(args.output_csv_file, 'w')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['asin', 'title', 'description', 'model_name', 'question'])
    all_rows = []
    asins_so_far = defaultdict(list)
    with open(args.previous_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            if row['model_name'] not in model_list:
                continue
            if asin not in asins_so_far[row['model_name']]:
                asins_so_far[row['model_name']].append(asin)
            else:
                print 'Duplicate asin %s in %s' % (asin, row['model_name'])
                continue
            all_rows.append([row['asin'], row['title'], row['description'], row['model_name'], row['question']])
    random.shuffle(all_rows)
    for row in all_rows:
        writer.writerow(row)
    csv_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--previous_csv_file", type=str)
    argparser.add_argument("--output_csv_file", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


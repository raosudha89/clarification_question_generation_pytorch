import argparse
import gzip
import nltk
import pdb
import sys
from collections import defaultdict
import csv
import random


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def read_model_outputs(model_fname):
    i = 0
    model_outputs = {}
    model_file = open(model_fname, 'r')
    test_ids = [line.strip('\n') for line in open(model_fname+'.ids', 'r')]
    for line in model_file.readlines():
        model_outputs[test_ids[i]] = line.strip('\n').replace(' <unk>', '').replace(' <EOS>', '')
        i += 1
    return model_outputs


def main(args):
    titles = {}
    descriptions = {}
    seq2seq_model_outs = read_model_outputs(args.seq2seq_model_fname)
    with open(args.previous_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            titles[asin] = row['title']
            descriptions[asin] = row['description']

    csv_file = open(args.output_csv_file, 'w')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['asin', 'title', 'description', 'model_name', 'question'])
    all_rows = []
    for asin in seq2seq_model_outs.keys():
        if asin not in titles:
            continue
        title = titles[asin]
        description = descriptions[asin]
        seq2seq_question = seq2seq_model_outs[asin]
        all_rows.append([asin, title, description, args.seq2seq_model_name, seq2seq_question])
    random.shuffle(all_rows)
    for row in all_rows:
        writer.writerow(row)
    csv_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--previous_csv_file", type=str)
    argparser.add_argument("--output_csv_file", type=str)
    argparser.add_argument("--seq2seq_model_fname", type=str)
    argparser.add_argument("--seq2seq_model_name", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


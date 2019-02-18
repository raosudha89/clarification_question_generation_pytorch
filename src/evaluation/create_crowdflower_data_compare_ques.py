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


def main(args):
    titles = {}
    descriptions = {}

    previous_asins = []
    with open(args.previous_csv_file_v1) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            previous_asins.append(asin)
    with open(args.previous_csv_file_v2) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            previous_asins.append(asin)
    with open(args.previous_csv_file_v3) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            previous_asins.append(asin)
    with open(args.previous_csv_file_v4) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            previous_asins.append(asin)

    train_asins = [line.strip('\n') for line in open(args.train_asins, 'r').readlines()]

    for v in parse(args.metadata_fname):
        asin = v['asin']
        if asin not in train_asins:
            continue
        if asin in previous_asins:
            continue
        title = v['title']
        description = v['description']
        length = len(description.split())
        if length >= 100 or length < 10 or len(title.split()) == length:
            continue
        titles[asin] = title
        descriptions[asin] = description

    questions = defaultdict(list)
    for v in parse(args.qa_data_fname):
        asin = v['asin']
        if asin not in train_asins:
            continue
        if asin in previous_asins:
            continue
        if asin not in descriptions:
            continue
        question = ' '.join(nltk.sent_tokenize(v['question'])).lower()
        questions[asin].append(question)

    csv_file = open(args.csv_file, 'w')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['asin', 'title', 'description', 'question_a', 'question_b'])
    all_rows = []
    max_count = 200
    i = 0
    for asin in titles.keys():
        if asin not in train_asins:
            continue
        if asin in previous_asins:
            continue
        title = titles[asin]
        description = descriptions[asin]
        random.shuffle(questions[asin])
        for k in range(len(questions[asin])):
            if k == 6 or k == len(questions[asin])-1:
                all_rows.append([asin, title, description, questions[asin][k], questions[asin][0]])
                print asin, k+1
                break
            else:
                all_rows.append([asin, title, description, questions[asin][k], questions[asin][k + 1]])
        i += 1
        if i >= max_count:
            break
    random.shuffle(all_rows)
    for row in all_rows:
        writer.writerow(row)
    csv_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--qa_data_fname", type = str)
    argparser.add_argument("--metadata_fname", type = str)
    argparser.add_argument("--csv_file", type=str)
    argparser.add_argument("--train_asins", type=str)
    argparser.add_argument("--previous_csv_file_v1", type=str)
    argparser.add_argument("--previous_csv_file_v2", type=str)
    argparser.add_argument("--previous_csv_file_v3", type=str)
    argparser.add_argument("--previous_csv_file_v4", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


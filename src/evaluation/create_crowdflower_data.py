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
    lucene_model_outs = read_model_outputs(args.lucene_model_fname)
    seq2seq_model_outs = read_model_outputs(args.seq2seq_model_fname)
    rl_model_outs = read_model_outputs(args.rl_model_fname)
    gan_model_outs = read_model_outputs(args.gan_model_fname)

    prev_batch_asins = []
    with open(args.prev_batch_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asin = row['asin']
            prev_batch_asins.append(asin)

    for v in parse(args.metadata_fname):
        asin = v['asin']
        if asin in prev_batch_asins:
            continue
        if asin not in lucene_model_outs.keys():
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
        if asin in prev_batch_asins:
            continue
        if asin not in lucene_model_outs.keys():
            continue
        if asin not in descriptions:
            continue
        question = ' '.join(nltk.sent_tokenize(v['question'])).lower()
        questions[asin].append(question)

    csv_file = open(args.csv_file, 'w')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['asin', 'title', 'description', 'model_name', 'question'])
    all_rows = []
    max_count = 100
    i = 0
    for asin in lucene_model_outs.keys():
        if asin in prev_batch_asins:
            continue
        if asin not in titles:
            continue
        title = titles[asin]
        description = descriptions[asin]
        ref_question = random.choice(questions[asin])
        lucene_question = lucene_model_outs[asin]
        if lucene_question == '':
            print 'Found empty line in lucene'
            continue
        seq2seq_question = seq2seq_model_outs[asin]
        rl_question = rl_model_outs[asin]
        gan_question = gan_model_outs[asin]
        all_rows.append([asin, title, description, 'ref', ref_question])
        all_rows.append([asin, title, description, args.lucene_model_name, lucene_question])
        all_rows.append([asin, title, description, args.seq2seq_model_name, seq2seq_question])
        all_rows.append([asin, title, description, args.rl_model_name, rl_question])
        all_rows.append([asin, title, description, args.gan_model_name, gan_question])
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
    argparser.add_argument("--prev_batch_csv_file", type=str)
    argparser.add_argument("--csv_file", type=str)
    argparser.add_argument("--lucene_model_fname", type=str)
    argparser.add_argument("--lucene_model_name", type=str)
    argparser.add_argument("--seq2seq_model_fname", type=str)
    argparser.add_argument("--seq2seq_model_name", type=str)
    argparser.add_argument("--rl_model_fname", type=str)
    argparser.add_argument("--rl_model_name", type=str)
    argparser.add_argument("--gan_model_fname", type=str)
    argparser.add_argument("--gan_model_name", type=str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


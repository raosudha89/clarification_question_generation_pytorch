import argparse
import gzip
import nltk
import pdb
import sys, os

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def main(args):
    titles = {}
    descriptions = {}
    related_products = {}
    total_meta = 0
    meta = 0
    for v in parse(args.metadata_fname):
        total_meta += 1
        if 'description' not in v or 'title' not in v:
            continue
        titles[v['asin']] = ' '.join(nltk.word_tokenize(v['title']))
        description = ''
        for sent in nltk.sent_tokenize(v['description']):
            description += ' '.join(nltk.word_tokenize(sent)) + ' '
        descriptions[v['asin']] = description
        meta += 1

    src_seq_file = open(args.qa_data_fname[:-8]+'.asin.ordered.src', 'w')
    tgt_seq_file = open(args.qa_data_fname[:-8]+'.asin.ordered.tgt', 'w')
    #src_seq_file = open(args.qa_data_fname[:-8]+'.withmeta.src', 'w')
    #tgt_seq_file = open(args.qa_data_fname[:-8]+'.withmeta.tgt', 'w')
    #src_seq_file = open(args.qa_data_fname[:-8]+'.viewed.src', 'w')
    #tgt_seq_file = open(args.qa_data_fname[:-8]+'.viewed.tgt', 'w')
    asin_order_file = open(args.qa_data_fname[:-8]+'.asin.order', 'w')
    questions = {}
    answers = {}
    total_ques = 0
    ques = 0
    for v in parse(args.qa_data_fname):
        asin = v['asin']
        questions[v['asin']] = v['question']
        answers[v['asin']] = v['answer']
        total_ques += 1
        if asin not in descriptions:
            continue
        asin_order_file.write(asin+'\n')
        src = titles[asin] + ' . ' + descriptions[asin] + ' <EOP> '
        src_seq_file.write(src+'\n')
        tgt = ' '.join(nltk.word_tokenize(questions[asin]))
        tgt_seq_file.write(tgt+'\n')
        ques += 1
    print 'Total meta %d' % total_meta
    print 'Meta %d' % meta
    print 'Total ques %d' % total_ques
    print 'Ques %d' % ques
    src_seq_file.close()
    tgt_seq_file.close()
    asin_order_file.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--qa_data_fname", type = str)
    argparser.add_argument("--metadata_fname", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


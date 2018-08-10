import argparse
import gzip
import nltk
import pdb
import sys, os
import re
from collections import defaultdict

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

exception_chars = ['|', '/', '\\', '-', '(', ')', '!', ':', ';', '<', '>']

def preprocess(text):
    text = text.replace('|', ' ')
    text = text.replace('/', ' ')
    text = text.replace('\\', ' ')
    text = text.lower()
    #text = re.sub(r'\W+', ' ', text)
    ret_text = ''
    for sent in nltk.sent_tokenize(text):
        ret_text += ' '.join(nltk.word_tokenize(sent)) + ' '
    return ret_text

def main(args):
    brand_counts = defaultdict(int)
    for v in parse(args.metadata_fname):
        if 'description' not in v or 'title' not in v or 'brand' not in v:
            continue
        brand_counts[v['brand']] += 1
    low_ct_brands = []
    for brand, ct in brand_counts.iteritems():
        if ct < 100:
            low_ct_brands.append(brand)
    products = {}
    for v in parse(args.metadata_fname):
        if 'description' not in v or 'title' not in v or 'brand' not in v:
            continue
        if v['brand'] not in low_ct_brands:
            continue
        asin = v['asin']
        title = preprocess(v['title'])
        description = preprocess(v['description'])
        product = title + ' . ' + description
        products[asin] = product

    question_list = {}
    print 'Creating docs'
    for v in parse(args.qa_data_fname):
        asin = v['asin']
        if asin not in products:
            continue
        print asin        
        if asin not in question_list:
            f = open(os.path.join(args.product_dir, asin + '.txt'), 'w')
            f.write(products[asin])
            f.close()
            question_list[asin] = []
        question = preprocess(v['question'])
        question_list[asin].append(question)
        f = open(os.path.join(args.question_dir, asin + '_' + str(len(question_list[asin])) + '.txt'), 'w')
        f.write(question)
        f.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--qa_data_fname", type = str)
    argparser.add_argument("--metadata_fname", type = str)
    argparser.add_argument("--product_dir", type = str)
    argparser.add_argument("--question_dir", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)


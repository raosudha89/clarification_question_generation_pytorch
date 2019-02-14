import argparse
import sys, os
from collections import defaultdict
import csv
import math
import pdb
import random
import gzip


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def get_brand_info(metadata_fname):
    brand_info = {}
    for v in parse(metadata_fname):
        if 'description' not in v or 'title' not in v:
            continue
        asin = v['asin']
        if 'brand' not in v.keys():
            brand_info[asin] = None
        else:
            brand_info[asin] = v['brand']
    return brand_info


def create_lucene_preds(ids, args, quess, sim_prod):
    pred_file = open(args.lucene_pred_fname, 'w')
    for test_id in ids:
        prod_id = test_id.split('_')[0]
        choices = []
        for i in range(min(len(sim_prod[prod_id]), 3)):
            choices += quess[sim_prod[prod_id][i]][:3]
        if len(choices) == 0:
            pred_file.write('\n')
        else:
            pred_ques = random.choice(choices)
            pred_file.write(pred_ques+'\n')
    pred_file.close()


def get_sim_docs(sim_docs_filename, brand_info):
    sim_docs_file = open(sim_docs_filename, 'r')
    sim_docs = {}
    for line in sim_docs_file.readlines():
        parts = line.split()
        #sim_docs[parts[0]] = parts[1:]
        asin = parts[0]
        sim_docs[asin] = []
        if len(parts[1:]) == 0:
            continue
        for prod_id in parts[2:]:
            if brand_info[prod_id] and (brand_info[prod_id] != brand_info[asin]):
                sim_docs[asin].append(prod_id)
        if len(sim_docs[asin]) == 0:
            sim_docs[asin] = parts[10:13]
        if len(sim_docs[asin]) == 0:
            pdb.set_trace()
    return sim_docs


def read_data(args):
    print("Reading lines...")
    quess = {}
    test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]
    print 'No. of test ids: %d' % len(test_ids)
    quess_rand = defaultdict(list)
    for fname in os.listdir(args.ques_dir):
        with open(os.path.join(args.ques_dir, fname), 'r') as f:
            ques_id = fname[:-4]
            asin, q_no = ques_id.split('_')
            ques = f.readline().strip('\n')
            quess_rand[asin].append((ques, q_no))

    for asin in quess_rand:
        quess[asin] = [None]*len(quess_rand[asin])
        for (ques, q_no) in quess_rand[asin]:
            q_no = int(q_no)-1
            quess[asin][q_no] = ques

    brand_info = get_brand_info(args.metadata_fname)
    sim_prod = get_sim_docs(args.sim_prod_fname, brand_info)    
    create_lucene_preds(test_ids, args, quess, sim_prod)    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ques_dir", type = str)
    argparser.add_argument("--sim_prod_fname", type = str)
    argparser.add_argument("--test_ids_file", type = str)
    argparser.add_argument("--lucene_pred_fname", type = str)
    argparser.add_argument("--metadata_fname", type = str)
    args = argparser.parse_args()
    print args
    print ""
    read_data(args)

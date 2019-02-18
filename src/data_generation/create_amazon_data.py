import argparse
import sys, os
from collections import defaultdict
import csv
import math
import pdb
import gzip

MAX_POST_LEN=200
MAX_QUES_LEN=40
#MIN_TFIDF=30
MIN_TFIDF=2
#MIN_QUES_TFIDF=3000
#MIN_QUES_TFIDF=1000
MIN_QUES_TF=200
MAX_QUES_TFIDF=10
#MAX_QUES_TFIDF=8.5

def write_to_file(ids, args, prods, quess, template_quess, sim_prod, sim_ques, split):
    suffix = ''
    if args.nocontext:
        suffix += '_nocontext'
    if args.simqs:
        suffix += '_simqs'
    if args.candqs:
        suffix += '_candqs'
    if args.template:
        suffix += '_template'
    if args.onlycontext:
        suffix += '_onlycontext'
    suffix += '.txt'

    if split == 'train':
        src_file = open(args.train_src_fname+suffix, 'w')
        tgt_file = open(args.train_tgt_fname+suffix, 'w')
        ids_file = open(args.train_tgt_fname+suffix+'.ids', 'w')
    elif split == 'tune':
        src_file = open(args.tune_src_fname+suffix, 'w')
        tgt_file = open(args.tune_tgt_fname+suffix, 'w')
        ids_file = open(args.tune_tgt_fname+suffix+'.ids', 'w')
    if split == 'test':
        src_file = open(args.test_src_fname+suffix, 'w')
        tgt_file = open(args.test_tgt_fname+suffix, 'w')
        ids_file = open(args.test_tgt_fname+suffix+'.ids', 'w')

    for prod_id in ids:
        if args.candqs:
            if len(sim_prod[prod_id]) < 4:
                print prod_id
                continue
        for j in range(len(quess[prod_id])):
            src_line = ''
            if not args.nocontext:
                src_line += prods[prod_id]+' <EOP> '
            ques_id = prod_id+'_'+str(j+1)
            if args.simqs:
                if ques_id not in sim_ques or len(sim_ques[ques_id]) < 4:
                    break
            ids_file.write(ques_id+'\n')
            for k in range(1, 4):
                if args.candqs:
                    if args.template:
                        src_line += template_quess[sim_prod[prod_id][k]][(j)%len(template_quess[sim_prod[prod_id][k]])] + ' <EOQ> '
                    else:
                        src_line += quess[sim_prod[prod_id][k]][(j)%len(quess[sim_prod[prod_id][k]])] + ' <EOQ> '
                if args.simqs:
                    sim_prod_id, sim_q_no = sim_ques[ques_id][k].split('_')
                    sim_q_no = int(sim_q_no)-1
                    if args.template:
                        src_line += template_quess[sim_prod_id][sim_q_no] + ' <EOQ> '
                    else:
                        src_line += quess[sim_prod_id][sim_q_no] + ' <EOQ> '
            src_file.write(src_line+'\n')
            tgt_file.write(quess[prod_id][j]+'\n')
    src_file.close()
    tgt_file.close()
    ids_file.close()

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

def get_sim_prods(sim_prods_filename, brand_info):
    sim_prods_file = open(sim_prods_filename, 'r')
    sim_prods = {}
    for line in sim_prods_file.readlines():
        parts = line.split()
        asin = parts[0]
        sim_prods[asin] = []
        for prod_id in parts[2:]:
            if brand_info[prod_id] and (brand_info[prod_id] != brand_info[asin]):
                sim_prods[asin].append(prod_id)
        if len(sim_prods[asin]) == 0:
            sim_prods[asin] = parts[10:13]
    return sim_prods

def get_sim_quess(sim_quess_filename, brand_info):
    sim_quess_file = open(sim_quess_filename, 'r')
    sim_quess = {}
    for line in sim_quess_file.readlines():
        parts = line.split()
        asin = parts[0].split('_')[0]
        sim_prods = [p.split('_')[0] for p in parts[2:]]
        sim_quess_ids = parts[2:]
        sim_quess[parts[0]] = []
        for i, prod_id in enumerate(sim_prods):
            if brand_info[prod_id] and (brand_info[prod_id] != brand_info[asin]):
                sim_quess[parts[0]].append(sim_quess_ids[i])
        if len(sim_quess[parts[0]]) == 0:
            sim_quess[parts[0]] = parts[10:13]
            print 'No new lucene simqs %s' % parts[0]
    return sim_quess

def trim_by_len(s, max_len):
    s = s.lower().strip()
    words = s.split()
    s = ' '.join(words[:max_len])
    return s

def trim_by_tfidf(prods, p_tf, p_idf):
    for prod_id in prods:
        prod = []
        words = prods[prod_id].split()
        for w in words:
            tf = words.count(w)
            #if p_tf[w]*p_idf[w] >= MIN_TFIDF:
            if tf*p_idf[w] >= MIN_TFIDF:
                prod.append(w)
            if len(prod) >= MAX_POST_LEN:
                break
        prods[prod_id] = ' '.join(prod)
    return prods

def has_number(string):
    for char in string:
        if char.isdigit():
            return True
    return False


def template_by_tfidf(quess, q_tf, q_idf):
    template_quess = {}
    for prod_id in quess:
        for ques in quess[prod_id]:
            template_ques = []
            words = ques.split()
            #print words
            for w in words:
                tf = words.count(w)
                #if q_tf[w]*q_idf[w] >= MIN_QUES_TFIDF or w == '?':
                #if has_number(w) or tf*q_idf[w] > MAX_QUES_TFIDF:
                #if has_number(w) or q_idf[w] > MAX_QUES_TFIDF:
                if has_number(w) or q_tf[w] < MIN_QUES_TF:
                    #print w
                    template_ques.append('<BLANK>')
                else:
                    template_ques.append(w)
            #pdb.set_trace()
            if prod_id not in template_quess:
                template_quess[prod_id] = []
            template_quess[prod_id].append(' '.join(template_ques))
    return template_quess


def read_data(args):
    print("Reading lines...")
    prods = {}
    quess = {}
    p_tf = defaultdict(int)
    p_idf = defaultdict(int)
    N = 0
    for fname in os.listdir(args.prod_dir):
        with open(os.path.join(args.prod_dir, fname), 'r') as f:
            asin = fname[:-4]
            prod = f.readline().strip('\n')
            for w in prod.split():
                p_tf[w] += 1
            for w in set(prod.split()):
                p_idf[w] += 1    
            prods[asin] = prod
            N += 1

    for w in p_idf:
        p_idf[w] = math.log(N*1.0/p_idf[w])

    prods = trim_by_tfidf(prods, p_tf, p_idf)
    if os.path.isfile(args.train_ids_file):
        train_ids = [train_id.strip('\n') for train_id in open(args.train_ids_file, 'r').readlines()]
        tune_ids = [tune_id.strip('\n') for tune_id in open(args.tune_ids_file, 'r').readlines()]
        test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]
    else:
        ids = prods.keys()
        N = len(ids)
        train_ids = ids[:int(N*0.8)]
        tune_ids = ids[int(N*0.8):int(N*0.9)]
        test_ids = ids[int(N*0.9):]
        with open(args.train_ids_file, 'w') as f:
            for train_id in train_ids:
                f.write(train_id+'\n')
        with open(args.tune_ids_file, 'w') as f:
            for tune_id in tune_ids:
                f.write(tune_id+'\n')
        with open(args.test_ids_file, 'w') as f:
            for test_id in test_ids:
                f.write(test_id+'\n')
    
    N = 0
    q_tf = defaultdict(int)
    q_idf = defaultdict(int)
    quess_rand = defaultdict(list)
    for fname in os.listdir(args.ques_dir):
        with open(os.path.join(args.ques_dir, fname), 'r') as f:
            ques_id = fname[:-4]
            asin, q_no = ques_id.split('_')
            ques = f.readline().strip('\n')
            for w in ques.split():
                q_tf[w] += 1
            for w in set(ques.split()):
                q_idf[w] += 1
            quess_rand[asin].append((ques, q_no))
            N += 1

    for asin in quess_rand:
        quess[asin] = [None]*len(quess_rand[asin])
        for (ques, q_no) in quess_rand[asin]:
            q_no = int(q_no)-1
            quess[asin][q_no] = ques

    for w in q_idf:
        q_idf[w] = math.log(N*1.0/q_idf[w])

    brand_info = get_brand_info(args.metadata_fname)
    sim_prod = get_sim_prods(args.sim_prod_fname, brand_info)
    if args.simqs:
        sim_ques = get_sim_quess(args.sim_ques_fname, brand_info)
    else:
        sim_ques = None

    if args.template:
        template_quess = template_by_tfidf(quess, q_tf, q_idf)    
        write_to_file(train_ids, args, prods, quess, template_quess, sim_prod, sim_ques, split='train')
        write_to_file(tune_ids, args, prods, quess, template_quess, sim_prod, sim_ques, split='tune')    
        write_to_file(test_ids, args, prods, quess, template_quess, sim_prod, sim_ques, split='test')    
    else:
        template_quess = None
        write_to_file(train_ids, args, prods, quess, template_quess, sim_prod, sim_ques, split='train')
        write_to_file(tune_ids, args, prods, quess, template_quess, sim_prod, sim_ques, split='tune')    
        write_to_file(test_ids, args, prods, quess, template_quess, sim_prod, sim_ques, split='test')    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--prod_dir", type = str)
    argparser.add_argument("--ques_dir", type = str)
    argparser.add_argument("--metadata_fname", type = str)
    argparser.add_argument("--sim_prod_fname", type = str)
    argparser.add_argument("--sim_ques_fname", type = str)
    argparser.add_argument("--train_ids_file", type = str)
    argparser.add_argument("--train_src_fname", type = str)
    argparser.add_argument("--train_tgt_fname", type = str)
    argparser.add_argument("--tune_ids_file", type = str)
    argparser.add_argument("--tune_src_fname", type = str)
    argparser.add_argument("--tune_tgt_fname", type = str)
    argparser.add_argument("--test_ids_file", type = str)
    argparser.add_argument("--test_src_fname", type = str)
    argparser.add_argument("--test_tgt_fname", type = str)
    argparser.add_argument("--candqs", type = bool)
    argparser.add_argument("--simqs", type = bool)
    argparser.add_argument("--template", type = bool)
    argparser.add_argument("--nocontext", type = bool)
    argparser.add_argument("--onlycontext", type = bool)
    args = argparser.parse_args()
    print args
    print ""
    read_data(args)
    

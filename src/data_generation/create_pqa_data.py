import argparse
import sys
from collections import defaultdict
import csv
import math
import nltk


MAX_POST_LEN = 100
MAX_QUES_LEN = 20
MAX_ANS_LEN = 20


def write_to_file(ids, args, posts, question_candidates, answer_candidates, split):

    if split == 'train':
        context_file = open(args.train_context_fname, 'w')
        question_file = open(args.train_question_fname, 'w')
        answer_file = open(args.train_answer_fname, 'w')
    elif split == 'tune':
        context_file = open(args.tune_context_fname, 'w')
        question_file = open(args.tune_question_fname, 'w')
        answer_file = open(args.tune_answer_fname, 'w')
    elif split == 'test':
        context_file = open(args.test_context_fname, 'w')
        question_file = open(args.test_question_fname, 'w')
        answer_file = open(args.test_answer_fname, 'w')

    for k, post_id in enumerate(ids):
        context_file.write(posts[post_id]+'\n')
        question_file.write(question_candidates[post_id][0]+'\n')
        answer_file.write(answer_candidates[post_id][0]+'\n')

    context_file.close()
    question_file.close()
    answer_file.close()


def trim_by_len(s, max_len):
    s = s.lower().strip()
    words = s.split()
    s = ' '.join(words[:max_len])
    return s


def trim_by_tfidf(posts, p_tf, p_idf):
    for post_id in posts:
        post = []
        words = posts[post_id].split()
        for w in words:
            tf = words.count(w)
            if tf*p_idf[w] >= MIN_TFIDF:
                post.append(w)
            if len(post) >= MAX_POST_LEN:
                break
        posts[post_id] = ' '.join(post)
    return posts


def read_data(args):
    print("Reading lines...")
    posts = {}
    question_candidates = {}
    answer_candidates = {}
    p_tf = defaultdict(int)
    p_idf = defaultdict(int)
    with open(args.post_data_tsvfile, 'rb') as tsvfile:
        post_reader = csv.reader(tsvfile, delimiter='\t')
        N = 0
        for row in post_reader:
            if N == 0:
                N += 1
                continue
            N += 1
            post_id, title, post = row
            post = title + ' ' + post
            post = post.lower().strip()
            for w in post.split():
                p_tf[w] += 1
            for w in set(post.split()):
                p_idf[w] += 1    
            posts[post_id] = post 

    for w in p_idf:
        p_idf[w] = math.log(N*1.0/p_idf[w])

    # for asin, post in posts.iteritems():
    #     posts[asin] = trim_by_len(post, MAX_POST_LEN)
    #posts = trim_by_tfidf(posts, p_tf, p_idf)
    N = 0
    with open(args.qa_data_tsvfile, 'rb') as tsvfile:
        qa_reader = csv.reader(tsvfile, delimiter='\t')
        i = 0
        for row in qa_reader:
            if i == 0:
                i += 1
                continue
            post_id, questions = row[0], row[1:11]
            answers = row[11:21]
            # questions = [trim_by_len(question, MAX_QUES_LEN) for question in questions]
            question_candidates[post_id] = questions
            # answers = [trim_by_len(answer, MAX_ANS_LEN) for answer in answers]
            answer_candidates[post_id] = answers

    train_ids = [train_id.strip('\n') for train_id in open(args.train_ids_file, 'r').readlines()]
    tune_ids = [tune_id.strip('\n') for tune_id in open(args.tune_ids_file, 'r').readlines()]
    test_ids = [test_id.strip('\n') for test_id in open(args.test_ids_file, 'r').readlines()]
    
    write_to_file(train_ids, args, posts, question_candidates, answer_candidates, 'train')    
    write_to_file(tune_ids, args, posts, question_candidates, answer_candidates, 'tune')    
    write_to_file(test_ids, args, posts, question_candidates, answer_candidates, 'test')    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--post_data_tsvfile", type = str)
    argparser.add_argument("--qa_data_tsvfile", type = str)
    argparser.add_argument("--train_ids_file", type = str)
    argparser.add_argument("--train_context_fname", type = str)
    argparser.add_argument("--train_question_fname", type = str)
    argparser.add_argument("--train_answer_fname", type=str)
    argparser.add_argument("--tune_ids_file", type = str)
    argparser.add_argument("--tune_context_fname", type = str)
    argparser.add_argument("--tune_question_fname", type = str)
    argparser.add_argument("--tune_answer_fname", type=str)
    argparser.add_argument("--test_ids_file", type = str)
    argparser.add_argument("--test_context_fname", type=str)
    argparser.add_argument("--test_question_fname", type = str)
    argparser.add_argument("--test_answer_fname", type = str)
    args = argparser.parse_args()
    print args
    print ""
    read_data(args)

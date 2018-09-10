import re
import csv
from constants import *
import unicodedata
from collections import defaultdict
import math


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s, max_len):
    #s = unicode_to_ascii(s.lower().strip())
    s = s.lower().strip()
    words = s.split()
    s = ' '.join(words[:max_len])
    return s


def get_context(line, max_post_len, max_ques_len):
    splits = line.split('<EOP>')
    context = splits[0]
    context = normalize_string(context, max_post_len) + ' <EOP>'
    if len(splits) > 1:
        sim_ques = splits[1].split('<EOQ>')
        for ques in sim_ques:
            ques = normalize_string(ques, max_ques_len)
            context += ques + ' <EOQ>'
    return context


def read_data(context_fname, question_fname, answer_fname, ids_fname,
              max_post_len, max_ques_len, max_ans_len, count=None):
    print("Reading lines...")
    data = []
    i = 0
    for line in open(context_fname, 'r').readlines():
        context = get_context(line, max_post_len, max_ques_len)
        data.append([context, None, None])
        i += 1
        if count and i == count:
            break

    i = 0
    for line in open(question_fname, 'r').readlines():
        question = normalize_string(line, max_ques_len)
        data[i][1] = question
        i += 1
        if count and i == count:
            break
    assert(i == len(data))

    i = 0
    for line in open(answer_fname, 'r').readlines():
        answer = normalize_string(line, max_ans_len)
        data[i][2] = answer
        i += 1
        if count and i == count:
            break
    assert(i == len(data))

    if ids_fname is not None:
        updated_data = []
        i = 0
        id_so_far = None
        for line in open(ids_fname, 'r').readlines():
            curr_id = line.strip('\n')
            if id_so_far != curr_id:
                updated_data.append(data[i])
                id_so_far = curr_id
            i += 1
            if count and i == count:
                break
        assert (i == len(data))
        return updated_data

    return data

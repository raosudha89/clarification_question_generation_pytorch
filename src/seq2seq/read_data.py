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
    is_specific, is_generic = False, False
    if '<specific>' in line:
        line = line.replace(' <specific>', '')
        is_specific = True
    if '<generic>' in line:
        line = line.replace(' <generic>', '')
        is_generic = True
    if is_specific or is_generic:
        context = normalize_string(line, max_post_len-2)  # one token space for specificity and another for EOS
    else:
        context = normalize_string(line, max_post_len-1)  
    if is_specific:
        context = '<specific> ' + context
    if is_generic:
        context += '<generic> ' + context
    return context


def read_data(context_fname, question_fname, answer_fname, ids_fname,
              max_post_len, max_ques_len, max_ans_len, count=None, mode='train'):
    if ids_fname is not None:
        ids = []
        for line in open(ids_fname, 'r').readlines():
            curr_id = line.strip('\n')
            ids.append(curr_id)

    print("Reading lines...")
    data = []
    i = 0
    for line in open(context_fname, 'r').readlines():
        context = get_context(line, max_post_len, max_ques_len)
        if ids_fname is not None:
            data.append([ids[i], context, None, None])
        else:
            data.append([None, context, None, None])
        i += 1
        if count and i == count:
            break

    i = 0
    for line in open(question_fname, 'r').readlines():
        question = normalize_string(line, max_ques_len-1)
        data[i][2] = question
        i += 1
        if count and i == count:
            break
    assert(i == len(data))

    if answer_fname is not None:
        i = 0
        for line in open(answer_fname, 'r').readlines():
            answer = normalize_string(line, max_ans_len-1)  # one token space for EOS
            data[i][3] = answer
            i += 1
            if count and i == count:
                break
        assert(i == len(data))

    if ids_fname is not None:
        updated_data = []
        i = 0
        if mode == 'test':
            max_per_id_count = 1
        else:
            max_per_id_count = 20
        data_ct_per_id = defaultdict(int)
        for curr_id in ids:
            data_ct_per_id[curr_id] += 1
            if data_ct_per_id[curr_id] <= max_per_id_count:
                updated_data.append(data[i])
            i += 1
            if count and i == count:
                break
        assert (i == len(data))
        return updated_data

    return data

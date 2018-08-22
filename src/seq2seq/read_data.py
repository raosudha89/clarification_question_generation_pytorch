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

def get_context(line):
	splits = line.split('<EOP>')
	context = splits[0]
	context = normalize_string(context, MAX_POST_LEN) + ' <EOP>'
	if len(splits) > 1:
		sim_ques = splits[1].split('<EOQ>')
		for ques in sim_ques:
			ques = normalize_string(context, MAX_QUES_LEN)
			context += ques + ' <EOQ>'
	return context

def read_data(context, question, answer, split):
	print("Reading lines...")
	data = []

	for line in open(context, 'r').readlines():
		context = get_context(line)
		data.append([context, None, None])

	i = 0
	for line in open(question, 'r').readlines():
		question = normalize_string(line, MAX_QUES_LEN)
		data[i][1] = question
		i += 1
	assert(i == len(data))

	i = 0
	for line in open(answer, 'r').readlines():
		answer = normalize_string(line, MAX_ANS_LEN)
		data[i][2] = answer
		i += 1
	assert(i == len(data))

	return data

import re
import csv
from constants import *
import unicodedata
from collections import defaultdict
import math

class Data:
	def __init__(self, name, tf=None, idf=None):
		self.name = name
		self.trimmed = False
		#self.word2index = {"UNK": 0, "PAD": 1, "SOS": 2, "EOS": 3}
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "UNK", 1: "PAD", 2: "SOS", 3: "EOS"}
		self.n_words = 4 # Count default tokens
		self.tf = tf
		self.idf = idf

	def index_words(self, sentence):
		for word in sentence.split(' '):
			self.index_word(word)

	def index_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	# Remove words below a certain count threshold
	def trim(self, min_count):
		if self.trimmed: return
		self.trimmed = True
		
		keep_words = []
		
		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words %s / %s = %.4f' % (
			len(keep_words), len(self.word2index), len(keep_words)*1.0 / len(self.word2index)
		))

		# Reinitialize dictionaries
		#self.word2index = {"UNK": 0, "PAD": 1, "SOS": 2, "EOS": 3}
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
		self.n_words = 3 # Count default tokens

		for word in keep_words:
			self.index_word(word)
	
	def trim_using_tfidf(self):
		if self.trimmed: return
		self.trimmed = True
		
		keep_words = []
		
		for w in self.word2count:
			if self.tf[w]*self.idf[w] >= MIN_TFIDF:
				keep_words.append(w)

		print('keep_words %s / %s = %.4f' % (
			len(keep_words), len(self.word2index), len(keep_words)*1.0 / len(self.word2index)
		))

		# Reinitialize dictionaries
		#self.word2index = {"UNK": 0, "PAD": 1, "SOS": 2, "EOS": 3}
		self.word2index = {}
		self.word2count = {}
		#self.index2word = {0: "UNK", 1: "PAD", 2: "SOS", 3: "EOS"}
		#self.n_words = 4 # Count default tokens
		self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
		self.n_words = 3 # Count default tokens

		for word in keep_words:
			self.index_word(word)

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
	if split == 'train':
		p_tf = defaultdict(int)
		p_idf = defaultdict(int)

	for line in open(context, 'r').readlines():
		context = get_context(line)
		data.append([context, None, None])
		if split == 'train':
			for w in context.split():
				p_tf[w] += 1
			for w in set(context.split()):
				p_idf[w] += 1
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

	if split == 'train':
		p_data = Data('post', p_tf, p_idf)
		q_data = Data('question')
		a_data = Data('answer')
		return p_data, q_data, a_data, data
	else:
		return data

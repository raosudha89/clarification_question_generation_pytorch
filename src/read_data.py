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

def read_data(post_data_tsv, qa_data_tsv, train_ids_file, test_ids_file):
	print("Reading lines...")
	posts = {}
	questions = {}
	p_tf = defaultdict(int)
	p_idf = defaultdict(int)
	with open(post_data_tsv, 'rb') as tsvfile:
		post_reader = csv.reader(tsvfile, delimiter='\t')
		i = 0
		for row in post_reader:
			if i == 0:
				i += 1
				continue
			i += 1
			post_id,title,post = row
			post = title + ' ' + post
			post = normalize_string(post, MAX_POST_LEN)
			for w in post.split():
				p_tf[w] += 1
			for w in set(post.split()):
				p_idf[w] += 1	
			posts[post_id] = post 

	for w in p_idf:
		p_idf[w] = math.log(i*1.0/p_idf[w])

	with open(qa_data_tsv, 'rb') as tsvfile:
		qa_reader = csv.reader(tsvfile, delimiter='\t')
		i = 0
		for row in qa_reader:
			if i == 0:
				i += 1
				continue
			post_id,question = row[0], row[1]
			question = normalize_string(question, MAX_QUES_LEN)
			questions[post_id] = question

	train_ids = [train_id.strip('\n') for train_id in open(train_ids_file, 'r').readlines()]
	test_ids = [test_id.strip('\n') for test_id in open(test_ids_file, 'r').readlines()]
	train_triples = []
	test_triples = []
	for post_id in questions:	
		if post_id in train_ids:
			train_triples.append([posts[post_id], questions[post_id]])
		if post_id in test_ids:
			test_triples.append([posts[post_id], questions[post_id]])

	p_data = Data('post', p_tf, p_idf)
	q_data = Data('question')

	#For debugging
	#train_triples = train_triples[:1000]
	#test_triples = test_triples[:200]

	return p_data, q_data, train_triples, test_triples


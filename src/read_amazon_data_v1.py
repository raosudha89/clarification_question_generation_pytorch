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
	for v in parse(args.metadata_fname):
		if 'description' not in v or 'title' not in v or 'related' not in v or 'also_bought' not in v['related']:
		#if 'description' not in v or 'title' not in v:
			continue
		titles[v['asin']] = ' '.join(nltk.word_tokenize(v['title']))
		description = ''
		for sent in nltk.sent_tokenize(v['description']):
			description += ' '.join(nltk.word_tokenize(sent)) + ' '
		descriptions[v['asin']] = description
		related_products[v['asin']] = v['related']['also_bought']		

	for fname in os.listdir(args.qa_data_dir):
		if fname[:3] != 'qa_' or fname[-8:] != '.json.gz':
			continue
		src_seq_file = open(os.path.join(args.qa_data_dir, fname[:-8]+'.src'), 'w')
		tgt_seq_file = open(os.path.join(args.qa_data_dir, fname[:-8]+'.tgt'), 'w')
		fpath = os.path.join(args.qa_data_dir, fname)
		questions = {}
		answers = {}
		for v in parse(fpath):
			questions[v['asin']] = v['question']
			answers[v['asin']] = v['answer']
		for asin in questions:
			if asin not in descriptions:
				continue
			src = titles[asin] + ' ' + descriptions[asin] + ' <EOP> '
			n = 0
			for rel_asin in related_products[asin]:
				if rel_asin not in questions:
					continue
				src += ' '.join(nltk.word_tokenize(questions[rel_asin])) + ' <EOQ> '
				n += 1
				if n == 3:
					break
			if n < 3:
				continue
			src_seq_file.write(src+'\n')
			tgt = ' '.join(nltk.word_tokenize(questions[asin]))
			tgt_seq_file.write(tgt+'\n')
		src_seq_file.close()
		tgt_seq_file.close()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--qa_data_dir", type = str)
	argparser.add_argument("--metadata_fname", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


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
	for v in parse(args.metadata_fname):
		if v['asin'] == args.prod_id:
			print v['description']
			pdb.set_trace()	

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--metadata_fname", type = str)
	argparser.add_argument("--prod_id", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


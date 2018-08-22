import argparse
import csv
import sys

def main(args):
	print("Reading lines...")
	output_file = open(args.output_data, 'a')
	with open(args.post_data_tsv, 'rb') as tsvfile:
		post_reader = csv.DictReader(tsvfile, delimiter='\t')
		for row in post_reader:
			post = row['title'] + ' ' + row['post']
			post = post.lower().strip()
			output_file.write(post+'\n')

	with open(args.qa_data_tsv, 'rb') as tsvfile:
		qa_reader = csv.DictReader(tsvfile, delimiter='\t')
		for row in qa_reader:
			ques = row['q1'].lower().strip()
			ans = row['a1'].lower().strip()
			output_file.write(ques+'\n')
			output_file.write(ans+'\n')

	output_file.close()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_data_tsv", type = str)
	argparser.add_argument("--qa_data_tsv", type = str)
	argparser.add_argument("--output_data", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)


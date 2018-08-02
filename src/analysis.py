import csv
import sys
from read_data import normalize_string
from constants import MAX_QUES_LEN

def read_tsv(qa_data_tsv):
	questions = []
	with open(qa_data_tsv, 'rb') as tsvfile:
		qa_reader = csv.reader(tsvfile, delimiter='\t')
		i = 0
		for row in qa_reader:
			if i == 0:
				i += 1
				continue
			post_id,question = row[0], row[1]
			question = normalize_string(question, MAX_QUES_LEN)
			questions.append(question)
	return questions

if __name__ == "__main__":
	output_file = open(sys.argv[1], 'r')
	qa_data_tsvfile = sys.argv[2]
	train_questions = read_tsv(qa_data_tsvfile)
	diff, total = 0, 0
	for line in output_file.readlines():
		line = line.strip('\n')
		if "('<'," == line[:5]:
			ques = line[7:-7]
			ques = ques.strip()
			total += 1
			is_same = False
			for train_ques in train_questions:
				if ques in train_ques:
					is_same = True
			#if ques not in train_questions:
			if not is_same:
				diff += 1
				print ques
	print "%d / %d = %f" % (diff, total, diff*1.0/total)

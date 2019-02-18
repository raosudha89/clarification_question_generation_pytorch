#!/bin/bash

#SBATCH --job-name=pqa_data_aus
#SBATCH --output=pqa_data_aus
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=askubuntu_unix_superuser

DATA_DIR=/fs/clip-amr/ranking_clarification_questions/data/$SITENAME
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

python $SCRIPT_DIR/create_pqa_data.py	--post_data_tsvfile $DATA_DIR/post_data.tsv \
										--qa_data_tsvfile $DATA_DIR/qa_data.tsv \
									--train_ids_file $DATA_DIR/train_ids \
									--tune_ids_file $DATA_DIR/tune_ids \
									--test_ids_file $DATA_DIR/test_ids \
									--train_context_fname $CQ_DATA_DIR/train_context.txt \
									--train_question_fname $CQ_DATA_DIR/train_question.txt \
									--train_answer_fname $CQ_DATA_DIR/train_answer.txt \
									--tune_context_fname $CQ_DATA_DIR/tune_context.txt \
									--tune_question_fname $CQ_DATA_DIR/tune_question.txt \
									--tune_answer_fname $CQ_DATA_DIR/tune_answer.txt \
									--test_context_fname $CQ_DATA_DIR/test_context.txt \
									--test_question_fname $CQ_DATA_DIR/test_question.txt \
									--test_answer_fname $CQ_DATA_DIR/test_answer.txt \

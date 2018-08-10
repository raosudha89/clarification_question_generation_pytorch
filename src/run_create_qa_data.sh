#!/bin/bash

#SITENAME=askubuntu.com
#SITENAME=unix.stackexchange.com
#SITENAME=superuser.com
DATA_DIR=/fs/clip-amr/ranking_clarification_questions/data/$SITENAME
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/question_answering/$SITENAME
SCRIPT_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/clarification_question_generation_pytorch/src

mkdir -p $CQ_DATA_DIR

python $SCRIPT_DIR/create_qa_data.py	--post_data_tsvfile $DATA_DIR/post_data.tsv \
										--qa_data_tsvfile $DATA_DIR/qa_data.tsv \
									--train_ids_file $DATA_DIR/train_ids \
									--tune_ids_file $DATA_DIR/tune_ids \
									--test_ids_file $DATA_DIR/test_ids \
									--train_src_fname $CQ_DATA_DIR/train_src \
									--train_tgt_fname $CQ_DATA_DIR/train_tgt \
									--tune_src_fname $CQ_DATA_DIR/tune_src \
									--tune_tgt_fname $CQ_DATA_DIR/tune_tgt \
									--test_src_fname $CQ_DATA_DIR/test_src \
									--test_tgt_fname $CQ_DATA_DIR/test_tgt \

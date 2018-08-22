#!/bin/bash

#SITENAME=askubuntu.com
#SITENAME=unix.stackexchange.com
SITENAME=superuser.com

CQ_DATA_DIR=/fs/clip-amr/ranking_clarification_questions/data/$SITENAME
OUT_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/embedding_generation

python $SCRIPT_DIR/extract_data.py	--post_data_tsv $CQ_DATA_DIR/post_data.tsv \
									--qa_data_tsv $CQ_DATA_DIR/qa_data.tsv \
									--output_data $OUT_DATA_DIR/${SITENAME}.data.txt


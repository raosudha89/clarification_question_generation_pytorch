#!/bin/bash

SITENAME=askubuntu_unix_superuser
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/lucene
DATA_DIR=/fs/clip-amr/ranking_clarification_questions/data/$SITENAME
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/data/lucene/$SITENAME

python $SCRIPT_DIR/create_stackexchange_lucene_baseline.py	--post_data_tsvfile $DATA_DIR/post_data.tsv \
                                            				--qa_data_tsvfile $DATA_DIR/qa_data.tsv \
                                            				--test_ids_file $DATA_DIR/test_ids \
                                            				--lucene_output_file $CQ_DATA_DIR/test_pred_question_lucene.txt \


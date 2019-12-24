#!/bin/bash

SITENAME=askubuntu_unix_superuser
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/lucene
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation/$SITENAME

python $SCRIPT_DIR/create_stackexchange_lucene_baseline.py	--post_data_tsvfile $CQ_DATA_DIR/post_data.tsv \
                                            				--qa_data_tsvfile $CQ_DATA_DIR/qa_data.tsv \
                                            				--test_ids_file $CQ_DATA_DIR/test_ids \
                                            				--lucene_output_file $CQ_DATA_DIR/test_pred_question_lucene.txt \


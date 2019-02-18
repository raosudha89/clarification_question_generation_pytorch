#!/bin/bash

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
OLD_CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation/data/amazon/$SITENAME
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/filter_amazon_data_byid.py 	--train_ids $CQ_DATA_DIR/train_asin.txt \
												--train_answer $CQ_DATA_DIR/train_answer.txt \
												--train_candqs_ids $OLD_CQ_DATA_DIR/train_tgt_candqs.txt.ids \
												--train_answer_candqs $CQ_DATA_DIR/train_answer_candqs.txt \


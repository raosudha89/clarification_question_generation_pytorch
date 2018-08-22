#!/bin/bash

#SBATCH --job-name=qa_data_Home_and_Kitchen
#SBATCH --output=qa_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/question_answering/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_amazon_qa_data.py 	--qa_data_fname $DATA_DIR/qa_${SITENAME}.json.gz \
												--metadata_fname $DATA_DIR/meta_${SITENAME}.json.gz \
												--train_src_fname $CQ_DATA_DIR/train_src \
												--train_tgt_fname $CQ_DATA_DIR/train_tgt \
												--tune_src_fname $CQ_DATA_DIR/tune_src \
												--tune_tgt_fname $CQ_DATA_DIR/tune_tgt \
												--test_src_fname $CQ_DATA_DIR/test_src \
												--test_tgt_fname $CQ_DATA_DIR/test_tgt \


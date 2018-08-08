#!/bin/bash

#SBATCH --job-name=utility_data_Home_and_Kitchen
#SBATCH --output=utility_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
QA_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/question_answering/data
UTILITY_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/utility/data
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation/src/utility

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/data_loader.py	--train_contexts_fname $QA_DATA_DIR/$SITENAME/train_contexts.txt \
									--train_answers_fname $QA_DATA_DIR/$SITENAME/train_answers.txt \
									--tune_contexts_fname $QA_DATA_DIR/$SITENAME/tune_contexts.txt \
									--tune_answers_fname $QA_DATA_DIR/$SITENAME/tune_answers.txt \
									--test_contexts_fname $QA_DATA_DIR/$SITENAME/test_contexts.txt \
									--test_answers_fname $QA_DATA_DIR/$SITENAME/test_answers.txt \
									--train_data $UTILITY_DATA_DIR/$SITENAME/train_data.p \
									--tune_data $UTILITY_DATA_DIR/$SITENAME/tune_data.p \
									--test_data $UTILITY_DATA_DIR/$SITENAME/test_data.p \
									--word_to_ix $UTILITY_DATA_DIR/$SITENAME/word_to_ix.p \
	


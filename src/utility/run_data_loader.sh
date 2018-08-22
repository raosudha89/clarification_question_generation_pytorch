#!/bin/bash

#SBATCH --job-name=utility_data_aus
#SBATCH --output=utility_data_aus
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

#SITENAME=askubuntu.com
#SITENAME=unix.stackexchange.com
SITENAME=superuser.com
#SITENAME=askubuntu_unix_superuser
#SITENAME=Home_and_Kitchen
QA_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/question_answering/$SITENAME
UTILITY_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/utility/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/utility

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/data_loader.py	--train_contexts_fname $QA_DATA_DIR/train_src \
									--train_answers_fname $QA_DATA_DIR/train_tgt \
									--tune_contexts_fname $QA_DATA_DIR/tune_src \
									--tune_answers_fname $QA_DATA_DIR/tune_tgt \
									--test_contexts_fname $QA_DATA_DIR/test_src \
									--test_answers_fname $QA_DATA_DIR/test_tgt \
									--train_data $UTILITY_DATA_DIR/train_data.p \
									--tune_data $UTILITY_DATA_DIR/tune_data.p \
									--test_data $UTILITY_DATA_DIR/test_data.p \
									#--word_to_ix $UTILITY_DATA_DIR/word_to_ix.p \
	


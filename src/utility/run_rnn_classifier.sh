#!/bin/bash

#SBATCH --job-name=utility_aus_fullmodel_80K
#SBATCH --output=utility_aus_fullmodel_80K
#SBATCH --qos=gpu-short
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=16g

#SITENAME=askubuntu.com
#SITENAME=unix.stackexchange.com
#SITENAME=superuser.com
SITENAME=askubuntu_unix_superuser
#SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
UTILITY_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/utility/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/utility
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME/200_5Kvocab
#EMB_DIR=/fs/clip-amr/ranking_clarification_questions/embeddings

source /fs/clip-amr/gpu_virtualenv/bin/activate
export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/rnn_classifier.py	--train_data $UTILITY_DATA_DIR/train_data.p \
										--tune_data $UTILITY_DATA_DIR/tune_data.p \
										--test_data $UTILITY_DATA_DIR/test_data.p \
										--word_embeddings $EMB_DIR/word_embeddings.p \
										--vocab $EMB_DIR/vocab.p \
										--cuda True \
	 


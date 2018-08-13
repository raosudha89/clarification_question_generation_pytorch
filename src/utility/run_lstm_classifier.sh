#!/bin/bash

#SBATCH --job-name=utility_aus
#SBATCH --output=utility_aus
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=askubuntu_unix_superuser
#SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
UTILITY_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/utility/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/utility
EMB_DIR=/fs/clip-amr/ranking_clarification_questions/embeddings

source /fs/clip-amr/gpu_virtualenv/bin/activate
export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/lstm_classifier.py	--train_data $UTILITY_DATA_DIR/train_data.p \
										--tune_data $UTILITY_DATA_DIR/tune_data.p \
										--test_data $UTILITY_DATA_DIR/test_data.p \
										--word_embeddings $EMB_DIR/word_embeddings.p \
										--vocab $EMB_DIR/vocab.p \
										--cuda True \
	 


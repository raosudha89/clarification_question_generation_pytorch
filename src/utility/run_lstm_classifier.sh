#!/bin/bash

#SBATCH --job-name=utility_Home_and_Kitchen
#SBATCH --output=utility_Home_and_Kitchen
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
UTILITY_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/utility/data
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation/src/utility
EMB_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/data/embeddings

source /fs/clip-amr/gpu_virtualenv/bin/activate
export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/lstm_classifier.py	--train_data $UTILITY_DATA_DIR/$SITENAME/train_data.p \
										--tune_data $UTILITY_DATA_DIR/$SITENAME/tune_data.p \
										--test_data $UTILITY_DATA_DIR/$SITENAME/test_data.p \
										--word_embeddings $EMB_DIR/$SITENAME/word_embeddings.p \
										--vocab $EMB_DIR/$SITENAME/vocab.p \
										--cuda True \
	 


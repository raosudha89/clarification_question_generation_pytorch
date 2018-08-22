#!/bin/bash

#SBATCH --job-name=qa_HK
#SBATCH --output=qa_HK
#SBATCH --qos=gpu-long
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64g

#SITENAME=askubuntu.com
#SITENAME=askubuntu_unix_superuser
SITENAME=Home_and_Kitchen

CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/question_answering/$SITENAME
EMB_DIR=/fs/clip-amr/question_generation/datasets/embeddings
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH=/cliphomes/raosudha/anaconda2/bin:$PATH

python $SCRIPT_DIR/main.py 	--train_src $CQ_DATA_DIR/train_src \
							--train_tgt $CQ_DATA_DIR/train_tgt \
							--test_src $CQ_DATA_DIR/test_src \
							--test_tgt $CQ_DATA_DIR/test_tgt \
							--test_pred $CQ_DATA_DIR/test_pred \
							#--word_vec_fname $EMB_DIR/vectors_200.txt			

#!/bin/bash

#SBATCH --job-name=qg_HK_candqs_300
#SBATCH --output=qg_HK_candqs_300
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64g

#SITENAME=askubuntu_unix_superuser
SITENAME=Home_and_Kitchen

#CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation/data/$SITENAME
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation/data/amazon/$SITENAME
EMB_DIR=/fs/clip-amr/question_generation/datasets/embeddings
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
SUFFIX=candqs

export PATH=/cliphomes/raosudha/anaconda2/bin:$PATH

python $SCRIPT_DIR/main.py 	--train_src $CQ_DATA_DIR/train_src_${SUFFIX}.txt \
							--train_tgt $CQ_DATA_DIR/train_tgt_${SUFFIX}.txt \
							--test_src $CQ_DATA_DIR/test_src_${SUFFIX}.txt \
							--test_tgt $CQ_DATA_DIR/test_tgt_${SUFFIX}.txt \
							--word_vec_fname $EMB_DIR/vectors_200.txt			

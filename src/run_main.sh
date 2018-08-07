#!/bin/bash

#SBATCH --job-name=qg_only_post_100
#SBATCH --output=qg_only_post_100
#SBATCH --qos=gpu-long
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64g

#DATA_DIR=/fs/clip-amr/question_generation/data_v9/unix.stackexchange.com
#DATA_DIR=/fs/clip-amr/question_generation/data_v9/superuser.com
DATA_DIR=/fs/clip-amr/question_generation/data_v9/askubuntu.com
OLD_DATA_DIR=/fs/clip-amr/question_generation/datasets/stackexchange_v9/askubuntu.com
EMB_DIR=/fs/clip-amr/question_generation/datasets/embeddings
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH=/cliphomes/raosudha/anaconda2/bin:$PATH

python $SCRIPT_DIR/main.py --post_data_tsvfile $DATA_DIR/post_data.tsv \
							--qa_data_tsvfile $DATA_DIR/qa_data.tsv \
							--train_ids_file $DATA_DIR/train_ids \
							--test_ids_file $DATA_DIR/test_ids \
							--word_vec_fname $EMB_DIR/vectors_200.txt			

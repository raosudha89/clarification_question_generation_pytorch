#!/bin/bash

#SBATCH --job-name=pqa_data_Home_and_Kitchen
#SBATCH --output=pqa_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_amazon_pqa_data.py 	--qa_data_fname $DATA_DIR/qa_${SITENAME}.json.gz \
												--metadata_fname $DATA_DIR/meta_${SITENAME}.json.gz \
												--train_post_fname $CQ_DATA_DIR/train_post.txt \
												--train_ques_fname $CQ_DATA_DIR/train_ques.txt \
												--train_ans_fname $CQ_DATA_DIR/train_ans.txt \
												--tune_post_fname $CQ_DATA_DIR/tune_post.txt \
												--tune_ques_fname $CQ_DATA_DIR/tune_ques.txt \
												--tune_ans_fname $CQ_DATA_DIR/tune_ans.txt \
												--test_post_fname $CQ_DATA_DIR/test_post.txt \
												--test_ques_fname $CQ_DATA_DIR/test_ques.txt \
												--test_ans_fname $CQ_DATA_DIR/test_ans.txt \


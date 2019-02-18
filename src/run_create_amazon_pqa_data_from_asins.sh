#!/bin/bash

#SBATCH --job-name=pqa_data_Home_and_Kitchen
#SBATCH --output=pqa_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_amazon_pqa_data_from_asins.py 	--qa_data_fname $DATA_DIR/qa_${SITENAME}.json.gz \
												--metadata_fname $DATA_DIR/meta_${SITENAME}.json.gz \
												--train_asin_fname $CQ_DATA_DIR/train_asin.txt \
												--train_ans_fname $CQ_DATA_DIR/train_ans.txt \
												--tune_asin_fname $CQ_DATA_DIR/tune_asin.txt \
												--tune_context_fname $CQ_DATA_DIR/tune_context.txt \
												--tune_ques_fname $CQ_DATA_DIR/tune_ques.txt \
												--tune_ans_fname $CQ_DATA_DIR/tune_ans.txt \
												--test_asin_fname $CQ_DATA_DIR/test_asin.txt \
												--test_ans_fname $CQ_DATA_DIR/test_ans.txt \


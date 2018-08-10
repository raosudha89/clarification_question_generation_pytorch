#!/bin/bash

#SBATCH --job-name=qa_data_Home_and_Kitchen
#SBATCH --output=qa_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
SCRATCH_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/question_answering
#SCRIPT_DIR=/fs/clip-amr/clarification_question_generation/src
SCRIPT_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/clarification_question_generation_pytorch/src

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_amazon_qa_data.py 	--qa_data_fname $DATA_DIR/qa_${SITENAME}.json.gz \
												--metadata_fname $DATA_DIR/meta_${SITENAME}.json.gz \
												--contexts_fname $SCRATCH_DATA_DIR/$SITENAME/contexts.txt \
												--answers_fname $SCRATCH_DATA_DIR/$SITENAME/answers.txt \

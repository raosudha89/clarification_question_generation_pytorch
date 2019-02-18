#!/bin/bash

#SBATCH --job-name=lucene_data_Home_and_Kitchen
#SBATCH --output=lucene_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
SCRATCH_DATA_DIR=/fs/clip-scratch/raosudha/amazon_qa
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation/src

rm -r $SCRATCH_DATA_DIR/${SITENAME}/prod_docs/
rm -r $SCRATCH_DATA_DIR/${SITENAME}/ques_docs/
mkdir -p $SCRATCH_DATA_DIR/${SITENAME}/prod_docs/
mkdir -p $SCRATCH_DATA_DIR/${SITENAME}/ques_docs/

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_lucene_amazon.py 	--qa_data_fname $DATA_DIR/qa_${SITENAME}.json.gz \
											--metadata_fname $DATA_DIR/meta_${SITENAME}.json.gz \
											--product_dir $SCRATCH_DATA_DIR/${SITENAME}/prod_docs/ \
											--question_dir $SCRATCH_DATA_DIR/${SITENAME}/ques_docs/

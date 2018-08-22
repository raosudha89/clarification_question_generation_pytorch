#!/bin/bash

#SBATCH --job-name=emb_data_Home_and_Kitchen
#SBATCH --output=emb_data_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
EMB_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/embedding_generation

python $SCRIPT_DIR/extract_amazon_data.py	--qa_data_fname $DATA_DIR/qa_${SITENAME}.json.gz \
											--metadata_fname $DATA_DIR/meta_${SITENAME}.json.gz \
											--output_fname $EMB_DATA_DIR/${SITENAME}_data.txt 
		

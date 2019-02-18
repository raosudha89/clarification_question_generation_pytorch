#!/bin/bash

#SBATCH --job-name=candqs_Home_and_Kitchen
#SBATCH --output=candqs_Home_and_Kitchen
#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
METADATA_DIR=/fs/clip-corpora/amazon_qa
#DATA_DIR=/fs/clip-corpora/amazon_qa/$SITENAME
DATA_DIR=/fs/clip-scratch/raosudha/amazon_qa/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation/src-opennmt
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/data/amazon/$SITENAME

mkdir $CQ_DATA_DIR

#mkdir /fs/clip-corpora/amazon_qa/$SITENAME
#mkdir /fs/clip-amr/clarification_question_generation/data/amazon/$SITENAME

python $SCRIPT_DIR/create_amazon_data.py 	--prod_dir $DATA_DIR/prod_docs \
											--ques_dir $DATA_DIR/ques_docs \
                                            --metadata_fname $METADATA_DIR/meta_${SITENAME}.json.gz \
											--sim_prod_fname $DATA_DIR/lucene_similar_prods.txt \
											--sim_ques_fname $DATA_DIR/lucene_similar_ques.txt \
											--train_src_fname $CQ_DATA_DIR/train_src \
											--train_tgt_fname $CQ_DATA_DIR/train_tgt \
											--tune_src_fname $CQ_DATA_DIR/tune_src \
											--tune_tgt_fname $CQ_DATA_DIR/tune_tgt \
											--test_src_fname $CQ_DATA_DIR/test_src \
											--test_tgt_fname $CQ_DATA_DIR/test_tgt \
											--train_ids_file $CQ_DATA_DIR/train_ids \
											--tune_ids_file $CQ_DATA_DIR/tune_ids \
											--test_ids_file $CQ_DATA_DIR/test_ids \
                                            --candqs True\
											#--onlycontext True \
											#--simqs True \
											#--template True \
											#--nocontext True \

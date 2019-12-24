#!/bin/bash

#SBATCH --job-name=lucene_Home_and_Kitchen_test
#SBATCH --output=lucene_Home_and_Kitchen_test
#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/lucene
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME

python $SCRIPT_DIR/create_amazon_lucene_baseline.py 	--ques_dir $CQ_DATA_DIR/ques_docs \
                                                        --sim_prod_fname $CQ_DATA_DIR/lucene_similar_prods.txt \
														--test_ids_file $CQ_DATA_DIR/blind_test_pred_ques.txt.seq2seq.epoch100.beam0.ids \
                                                        --lucene_pred_fname $CQ_DATA_DIR/blind_test_pred_question.lucene.txt \
                                                        --metadata_fname $CQ_DATA_DIR/meta_Home_and_Kitchen.json.gz \
											

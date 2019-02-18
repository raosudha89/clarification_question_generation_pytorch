#!/bin/bash

#SBATCH --job-name=create_amazon_multi_refs
#SBATCH --output=create_amazon_multi_refs
#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation/
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME

python $SCRIPT_DIR/create_amazon_multi_refs.py 	--ques_dir $DATA_DIR/ques_docs \
												--test_ids_file $CQ_DATA_DIR/blind_test_pred_question.txt.GAN_selfcritic_pred_ans_3perid.epoch8.len_norm.beam0.ids \
                                                --ref_prefix $CQ_DATA_DIR/test_ref \
												--test_context_file $CQ_DATA_DIR/test_context.txt

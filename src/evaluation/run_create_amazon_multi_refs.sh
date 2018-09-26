#!/bin/bash

#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation/
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME

python $SCRIPT_DIR/create_amazon_multi_refs.py 	--ques_dir $DATA_DIR/ques_docs \
                                                --test_ids_file $CQ_DATA_DIR/blind_test_pred_question.txt.seq2seq.len_norm.beam0.ids \
                                                --ref_prefix $CQ_DATA_DIR/test_ref

#!/bin/bash

SITENAME=Home_and_Kitchen
DATA_DIR=/fs/clip-corpora/amazon_qa
OLD_CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation/data/amazon/$SITENAME
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/filter_output_perid.py 	    --test_ids $CQ_DATA_DIR/tune_asin.txt \
                                                --test_output $CQ_DATA_DIR/test_pred_question.txt.pretrained_greedy \
												--test_output_perid $CQ_DATA_DIR/test_pred_question.txt.pretrained_greedy.perid \
												--batch_size 128 \
												# --max_per_id 3 \
												# --test_output $CQ_DATA_DIR/test_pred_question.txt.GAN_mixer_pred_ans_3perid.epoch8.beam0 \
												# --test_output_perid $CQ_DATA_DIR/test_pred_question.txt.GAN_mixer_pred_ans_3perid.epoch8.beam0.perid \
												# --test_output $CQ_DATA_DIR/test_pred_question.txt.pretrained_greedy \
												# --test_output_perid $CQ_DATA_DIR/test_pred_question.txt.pretrained_greedy.perid \


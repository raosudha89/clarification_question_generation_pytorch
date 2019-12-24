#!/bin/bash

#SBATCH --job-name=meteor
#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=4:00:00

METEOR=/fs/clip-software/user-supported/meteor-1.5
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/Home_and_Kitchen/
RESULTS_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/results/Home_and_Kitchen

TEST_SET=test_pred_ques.txt.seq2seq.epoch100.beam0.nounks

java -Xmx2G -jar $METEOR/meteor-1.5.jar $CQ_DATA_DIR/$TEST_SET $CQ_DATA_DIR/test_ref_combined \
										-l en -norm -r 10 \
									> $RESULTS_DIR/${TEST_SET}.meteor

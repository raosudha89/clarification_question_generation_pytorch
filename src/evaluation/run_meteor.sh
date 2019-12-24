#!/bin/bash

#SBATCH --job-name=meteor
#SBATCH --output=meteor
#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=4:00:00

SITENAME=askubuntu_unix_superuser

METEOR=/fs/clip-software/user-supported/meteor-1.5
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
RESULTS_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/results/$SITENAME

TEST_SET=test_pred_question.txt.RL_selfcritic.epoch8.hasrefs.nounks

java -Xmx2G -jar $METEOR/meteor-1.5.jar $CQ_DATA_DIR/$TEST_SET 	$CQ_DATA_DIR/test_ref_combined \
										-l en -norm -r 6 \
									> $RESULTS_DIR/${TEST_SET}.meteor

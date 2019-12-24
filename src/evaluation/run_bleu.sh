#!/bin/bash

SITENAME=Home_and_Kitchen
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
BLEU_SCRIPT=/fs/clip-software/user-supported/mosesdecoder/3.0/scripts/generic/multi-bleu.perl

#MODEL=mixer
#MODEL=selfcritic
#MODEL=mixer_noutil
#MODEL=selfcritic_noutil
#MODEL=mixer_pred_ans
#MODEL=selfcritic_pred_ans

$BLEU_SCRIPT $CQ_DATA_DIR/test_ref < \
								$CQ_DATA_DIR/GAN_test_pred_question.txt.${MODEL}.epoch10.hasrefs
								#$CQ_DATA_DIR/GAN_test_pred_question.txt${MODEL}.epoch10.hasrefs
								#$CQ_DATA_DIR/RL_test_pred_question_${MODEL}.txt.epoch16.hasrefs

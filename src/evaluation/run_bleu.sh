#!/bin/bash

SITENAME=Home_and_Kitchen
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
BLEU_SCRIPT=/fs/clip-software/user-supported/mosesdecoder/3.0/scripts/generic/multi-bleu.perl
REF_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/$SITENAME

#MODEL=mixer
#MODEL=selfcritic
#MODEL=mixer_noutil
#MODEL=selfcritic_noutil
#MODEL=mixer_pred_ans
#MODEL=selfcritic_pred_ans

$BLEU_SCRIPT $REF_DIR/test_ref < \
								$CQ_DATA_DIR/blind_test_pred_ques.txt.GAN_selfcritic_pred_ans_3perid_specificity.beam0
								#$CQ_DATA_DIR/RL_beam_decoder.txt.pretrained.beam0.hasrefs
								#$CQ_DATA_DIR/RL_beam_decoder_len_norm_wu.txt.pretrained.beam0.hasrefs
								#$CQ_DATA_DIR/GAN_test_pred_question.txt.${MODEL}.epoch10.beam0.hasrefs
								#$CQ_DATA_DIR/GAN_test_pred_question.txt${MODEL}.epoch10.beam0.hasrefs
								#$CQ_DATA_DIR/RL_test_pred_question_${MODEL}.txt.epoch16.beam0.hasrefs

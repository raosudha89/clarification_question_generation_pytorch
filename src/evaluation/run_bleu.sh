#!/bin/bash

SITENAME=Home_and_Kitchen
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
BLEU_SCRIPT=/fs/clip-software/user-supported/mosesdecoder/3.0/scripts/generic/multi-bleu.perl

$BLEU_SCRIPT $CQ_DATA_DIR/test_ref < $CQ_DATA_DIR/GAN_test_pred_question.txt.epoch8.hasrefs

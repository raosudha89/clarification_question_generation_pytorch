#!/bin/bash

DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/utility
UBUNTU=askubuntu.com
UNIX=unix.stackexchange.com
SUPERUSER=superuser.com
SCRIPTS_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/utility
SITE_NAME=askubuntu_unix_superuser

mkdir $DATA_DIR/$SITE_NAME

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/train_data.p \
										$DATA_DIR/$UNIX/train_data.p \
										$DATA_DIR/$SUPERUSER/train_data.p \
										$DATA_DIR/$SITE_NAME/train_data.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/tune_data.p \
										$DATA_DIR/$UNIX/tune_data.p \
										$DATA_DIR/$SUPERUSER/tune_data.p \
										$DATA_DIR/$SITE_NAME/tune_data.p

python $SCRIPTS_DIR/combine_pickle.py 	$DATA_DIR/$UBUNTU/test_data.p \
										$DATA_DIR/$UNIX/test_data.p \
										$DATA_DIR/$SUPERUSER/test_data.p \
										$DATA_DIR/$SITE_NAME/test_data.p


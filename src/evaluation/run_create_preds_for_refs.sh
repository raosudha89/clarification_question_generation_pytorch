#!/bin/bash

#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=05:00:00

SITENAME=askubuntu_unix_superuser
CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation

PRED_FILE=test_pred_question.txt.GAN_selfcritic_pred_ans_util_dis.epoch8

python $SCRIPT_DIR/create_preds_for_refs.py	--qa_data_tsvfile $CQ_DATA_DIR/qa_data.tsv \
  											--test_ids_file $CQ_DATA_DIR/test_ids \
										 	--human_annotations $CQ_DATA_DIR/human_annotations \
											--model_output_file $CQ_DATA_DIR/$PRED_FILE


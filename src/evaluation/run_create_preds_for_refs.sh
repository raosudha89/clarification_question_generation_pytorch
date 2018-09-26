#!/bin/bash

#SBATCH --qos=batch
#SBATCH --mem=4g
#SBATCH --time=05:00:00

SITENAME=askubuntu_unix_superuser
DATA_DIR=/fs/clip-amr/ranking_clarification_questions/data/$SITENAME
CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
UPWORK=/fs/clip-amr/question_generation/upwork/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation

python $SCRIPT_DIR/create_preds_for_refs.py	--qa_data_tsvfile $DATA_DIR/qa_data.tsv \
  											--test_ids_file $DATA_DIR/test_ids \
										 	--human_annotations $UPWORK/human_annotations \
											--model_output_file $CQ_DATA_DIR/test_pred_question.txt.mixer_pred_ans.epoch8.len_norm.beam0 \
											#--model_output_file $CQ_DATA_DIR/test_pred_question.txt.seq2seq.len_norm.beam0 \
											#--model_output_file $CQ_DATA_DIR/test_pred_question.txt.RL_mixer.epoch8.len_norm.beam0 \
											#--model_output_file $CQ_DATA_DIR/test_pred_question.txt.selfcritic_pred_ans.epoch8.len_norm.beam0 \
											#--model_output_file $CQ_DATA_DIR/test_question.txt \


#!/bin/bash

#SBATCH --job-name=joint_aus_emb100
#SBATCH --output=joint_aus_emb100
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64g

SITENAME=askubuntu_unix_superuser

CQ_DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/100
#EMB_DIR=/fs/clip-amr/ranking_clarification_questions/embeddings

export PATH=/cliphomes/raosudha/anaconda2/bin:$PATH

python $SCRIPT_DIR/joint_main.py	--train_context $CQ_DATA_DIR/train_context.txt \
									--train_question $CQ_DATA_DIR/train_question.txt \
									--train_answer $CQ_DATA_DIR/train_answer.txt \
									--tune_context $CQ_DATA_DIR/tune_context.txt \
									--tune_question $CQ_DATA_DIR/tune_question.txt \
									--tune_answer $CQ_DATA_DIR/tune_answer.txt \
									--test_context $CQ_DATA_DIR/test_context.txt \
									--test_question $CQ_DATA_DIR/test_question.txt \
									--test_answer $CQ_DATA_DIR/test_answer.txt \
									--test_pred_question $CQ_DATA_DIR/test_pred_question.txt \
									--test_pred_answer $CQ_DATA_DIR/test_pred_answer.txt \
									--word_embeddings $EMB_DIR/word_embeddings.p \
									--vocab $EMB_DIR/vocab.p \


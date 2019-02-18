#!/bin/bash

#SBATCH --job-name=spec_GAN_selfcritic_pred_ans_contd8
#SBATCH --output=spec_GAN_selfcritic_pred_ans_contd8
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=128g

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
#PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME/3perid_tobeginning
PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME/tobeginning
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
EMB_DIR=/fs/clip-amr/style_clarification_question_generation/embeddings/$SITENAME/200

#module load cuda/9.1.85

python $SCRIPT_DIR/GAN_main.py	--train_context $CQ_DATA_DIR/train_context_specificity.txt \
									--train_ques $CQ_DATA_DIR/train_ques.txt \
									--train_ans $CQ_DATA_DIR/train_ans.txt \
									--train_ids $CQ_DATA_DIR/train_asin.txt \
									--tune_context $CQ_DATA_DIR/tune_context_specificity.txt \
									--tune_ques $CQ_DATA_DIR/tune_ques.txt \
									--tune_ans $CQ_DATA_DIR/tune_ans.txt \
									--tune_ids $CQ_DATA_DIR/tune_asin.txt \
									--test_context $CQ_DATA_DIR/test_context_specificity.txt \
									--test_ques $CQ_DATA_DIR/test_ques.txt \
									--test_ans $CQ_DATA_DIR/test_ans.txt \
									--test_ids $CQ_DATA_DIR/test_asin.txt \
                                    --test_pred_ques $CQ_DATA_DIR/test_pred_ques.txt \
									--q_encoder_params $PARAMS_DIR/q_encoder_params.epoch65.GAN_selfcritic_pred_ans.epoch8 \
									--q_decoder_params $PARAMS_DIR/q_decoder_params.epoch65.GAN_selfcritic_pred_ans.epoch8 \
									--a_encoder_params $PARAMS_DIR/a_encoder_params.epoch60 \
									--a_decoder_params $PARAMS_DIR/a_decoder_params.epoch60 \
									--context_params $PARAMS_DIR/context_params.epoch10.GAN_selfcritic_pred_ans.epoch8 \
									--question_params $PARAMS_DIR/question_params.epoch10.GAN_selfcritic_pred_ans.epoch8 \
									--answer_params $PARAMS_DIR/answer_params.epoch10.GAN_selfcritic_pred_ans.epoch8 \
									--utility_params $PARAMS_DIR/utility_params.epoch10.GAN_selfcritic_pred_ans.epoch8 \
									--word_embeddings $EMB_DIR/word_embeddings.p \
									--vocab $EMB_DIR/vocab.p \
									--model GAN_selfcritic_pred_ans \
									--max_post_len 100 \
									--max_ques_len 20 \
									--max_ans_len 20 \
									--batch_size 64 \
									--n_epochs 20 \


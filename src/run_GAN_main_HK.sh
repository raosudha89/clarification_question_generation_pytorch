#!/bin/bash

#SBATCH --job-name=GAN_selfcritic_pred_ans_3perid_util_dis
#SBATCH --output=GAN_selfcritic_pred_ans_3perid_util_dis
#SBATCH --qos=gpu-long
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=128g

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
PARAMS_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME

#module load cuda/9.1.85

python $SCRIPT_DIR/GAN_main.py	--train_context $CQ_DATA_DIR/train_context.txt \
									--train_ques $CQ_DATA_DIR/train_ques.txt \
									--train_ans $CQ_DATA_DIR/train_ans.txt \
									--train_ids $CQ_DATA_DIR/train_asin.txt \
									--tune_context $CQ_DATA_DIR/tune_context.txt \
									--tune_ques $CQ_DATA_DIR/tune_ques.txt \
									--tune_ans $CQ_DATA_DIR/tune_ans.txt \
									--tune_ids $CQ_DATA_DIR/tune_asin.txt \
									--test_context $CQ_DATA_DIR/test_context.txt \
									--test_ques $CQ_DATA_DIR/test_ques.txt \
									--test_ans $CQ_DATA_DIR/test_ans.txt \
									--test_ids $CQ_DATA_DIR/test_asin.txt \
                                    --test_pred_ques $CQ_DATA_DIR/blind_test_pred_ques.txt \
									--q_encoder_params $PARAMS_DIR/q_encoder_params.epoch100 \
									--q_decoder_params $PARAMS_DIR/q_decoder_params.epoch100 \
									--a_encoder_params $PARAMS_DIR/a_encoder_params.epoch100 \
									--a_decoder_params $PARAMS_DIR/a_decoder_params.epoch100 \
									--context_params $PARAMS_DIR/context_params.epoch10 \
									--question_params $PARAMS_DIR/question_params.epoch10 \
									--answer_params $PARAMS_DIR/answer_params.epoch10 \
									--utility_params $PARAMS_DIR/utility_params.epoch10 \
									--word_embeddings $EMB_DIR/word_embeddings.p \
									--vocab $EMB_DIR/vocab.p \
									--model GAN_selfcritic_pred_ans_3perid_util_dis \
									--max_post_len 100 \
									--max_ques_len 20 \
									--max_ans_len 20 \
									--batch_size 64 \
									--n_epochs 20 \


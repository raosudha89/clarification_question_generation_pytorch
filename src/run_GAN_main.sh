#!/bin/bash

#SBATCH --job-name=GAN_aus_emb200_selfcritic_pred_ans
#SBATCH --output=GAN_aus_emb200_selfcritic_pred_ans
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64g

SITENAME=askubuntu_unix_superuser

CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME

python $SCRIPT_DIR/GAN_main.py    --train_context $CQ_DATA_DIR/train_context.txt \
                                    --train_ques $CQ_DATA_DIR/train_question.txt \
                                    --train_ans $CQ_DATA_DIR/train_answer.txt \
                                    --tune_context $CQ_DATA_DIR/tune_context.txt \
                                    --tune_ques $CQ_DATA_DIR/tune_question.txt \
                                    --tune_ans $CQ_DATA_DIR/tune_answer.txt \
                                    --test_context $CQ_DATA_DIR/test_context.txt \
                                    --test_ques $CQ_DATA_DIR/test_question.txt \
                                    --test_ans $CQ_DATA_DIR/test_answer.txt \
                                    --test_pred_ques $CQ_DATA_DIR/GAN_test_pred_question.txt \
                                    --q_encoder_params $CQ_DATA_DIR/q_encoder_params.epoch100 \
                                    --q_decoder_params $CQ_DATA_DIR/q_decoder_params.epoch100 \
                                    --a_encoder_params $CQ_DATA_DIR/a_encoder_params.epoch100 \
                                    --a_decoder_params $CQ_DATA_DIR/a_decoder_params.epoch100 \
                                    --context_params $CQ_DATA_DIR/context_params.epoch10 \
                                    --question_params $CQ_DATA_DIR/question_params.epoch10 \
                                    --answer_params $CQ_DATA_DIR/answer_params.epoch10 \
                                    --utility_params $CQ_DATA_DIR/utility_params.epoch10 \
                                    --word_embeddings $EMB_DIR/word_embeddings.p \
                                    --vocab $EMB_DIR/vocab.p \
									--model GAN_selfcritic_pred_ans \
                                    --max_post_len 100 \
                                    --max_ques_len 20 \
                                    --max_ans_len 20 \
                                    --batch_size 256 \
                                    --n_epochs 40 \


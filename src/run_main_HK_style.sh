#!/bin/bash

#SBATCH --job-name=pretrain_spec_ques_p100_q30_tobeginning_HK_emb200_style_emb
#SBATCH --output=pretrain_spec_ques_p100_q30_tobeginning_HK_emb200_style_emb
#SBATCH --qos=gpu-long
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64g

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME/tobeginning
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
#EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME/200
EMB_DIR=/fs/clip-amr/style_clarification_question_generation/embeddings/$SITENAME/200

export PATH=/cliphomes/raosudha/anaconda2/bin:$PATH

python $SCRIPT_DIR/main.py    --train_context $CQ_DATA_DIR/train_context_specificity.txt \
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
                                    --q_encoder_params $PARAMS_DIR/q_encoder_params_p100_q30_style_emb \
                                    --q_decoder_params $PARAMS_DIR/q_decoder_params_p100_q30_style_emb \
                                    --a_encoder_params $PARAMS_DIR/a_encoder_params \
                                    --a_decoder_params $PARAMS_DIR/a_decoder_params \
                                    --context_params $PARAMS_DIR/context_params \
                                    --question_params $PARAMS_DIR/question_params \
                                    --answer_params $PARAMS_DIR/answer_params \
                                    --utility_params $PARAMS_DIR/utility_params \
                                    --word_embeddings $EMB_DIR/word_embeddings.p \
                                    --vocab $EMB_DIR/vocab.p \
                                    --n_epochs 100 \
                                    --max_post_len 100 \
									--max_ques_len 30 \
									--max_ans_len 30 \
									--pretrain_ques True \
									#--pretrain_ans True \
									#--pretrain_util True \







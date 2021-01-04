#!/bin/bash

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=clarification_question_generation_pytorch/src
EMB_DIR=clarification_question_generation_pytorch/embeddings/$SITENAME

PARAMS_DIR=$CQ_DATA_DIR

python $SCRIPT_DIR/decode.py        --test_context $CQ_DATA_DIR/test_context.txt \
                                    --test_ques $CQ_DATA_DIR/test_ques.txt \
                                    --test_ans $CQ_DATA_DIR/test_ans.txt \
                                    --test_ids $CQ_DATA_DIR/test_asin.txt \
                                    --test_pred_ques $CQ_DATA_DIR/blind_test_pred_ques.txt \
                                    --q_encoder_params $PARAMS_DIR/q_encoder_params.epoch100.RL_selfcritic.epoch8 \
                                    --q_decoder_params $PARAMS_DIR/q_decoder_params.epoch100.RL_selfcritic.epoch8 \
                                    --word_embeddings $EMB_DIR/word_embeddings.p \
                                    --vocab $EMB_DIR/vocab.p \
                                    --model RL_selfcritic.epoch8 \
                                    --max_post_len 100 \
                                    --max_ques_len 20 \
                                    --max_ans_len 20 \
                                    --batch_size 128 \
                                    --beam True


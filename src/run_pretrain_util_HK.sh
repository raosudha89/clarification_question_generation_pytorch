#!/bin/bash

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=clarification_question_generation_pytorch/src
EMB_DIR=clarification_question_generation_pytorch/embeddings/$SITENAME

PARAMS_DIR=$CQ_DATA_DIR

python $SCRIPT_DIR/main.py    --train_context $CQ_DATA_DIR/train_context.txt \
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
                                    --q_encoder_params $PARAMS_DIR/q_encoder_params \
                                    --q_decoder_params $PARAMS_DIR/q_decoder_params \
                                    --a_encoder_params $PARAMS_DIR/a_encoder_params \
                                    --a_decoder_params $PARAMS_DIR/a_decoder_params \
                                    --context_params $PARAMS_DIR/context_params \
                                    --question_params $PARAMS_DIR/question_params \
                                    --answer_params $PARAMS_DIR/answer_params \
                                    --utility_params $PARAMS_DIR/utility_params \
                                    --word_embeddings $EMB_DIR/word_embeddings.p \
                                    --vocab $EMB_DIR/vocab.p \
                                    --n_epochs 10 \
                                    --max_post_len 100 \
                                    --max_ques_len 20 \
                                    --max_ans_len 20 \
                                    --pretrain_util True \

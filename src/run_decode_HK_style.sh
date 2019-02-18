#!/bin/bash

#SBATCH --job-name=HK_decode_seq2seq_tobeginning_tospecific_p100_q30_style_emb.epoch65
#SBATCH --output=HK_decode_seq2seq_tobeginning_tospecific_p100_q30_style_emb.epoch65
#SBATCH --qos=batch
#SBATCH --time=5:00:00
#SBATCH --mem=16g

SITENAME=Home_and_Kitchen

CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
#PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME/3perid
#PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME
#PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME/3perid_tobeginning
PARAMS_DIR=/fs/clip-scratch/raosudha/style_clarification_question_generation/$SITENAME/tobeginning
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
EMB_DIR=/fs/clip-amr/style_clarification_question_generation/embeddings/$SITENAME/200

python $SCRIPT_DIR/decode.py        --test_context $CQ_DATA_DIR/test_context_tospecific.txt \
									--test_ques $CQ_DATA_DIR/test_ques.txt \
									--test_ans $CQ_DATA_DIR/test_ans.txt \
									--test_ids $CQ_DATA_DIR/test_asin.txt \
									--test_pred_ques $CQ_DATA_DIR/blind_test_pred_ques.txt \
									--q_encoder_params $PARAMS_DIR/q_encoder_params_p100_q30_style_emb.epoch65 \
									--q_decoder_params $PARAMS_DIR/q_decoder_params_p100_q30_style_emb.epoch65 \
									--word_embeddings $EMB_DIR/word_embeddings.p \
									--vocab $EMB_DIR/vocab.p \
									--model seq2seq_tobeginning_tospecific_p100_q30_style_emb.epoch65 \
									--max_post_len 100 \
									--max_ques_len 30 \
									--max_ans_len 30 \
									--batch_size 128 \
									--n_epochs 40 \
									--beam True
									#--greedy True
									#--diverse_beam True \
									#--model seq2seq_tobeginning.beam.epoch65 \


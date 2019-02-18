#!/bin/bash

#SBATCH --job-name=aus_decode_emb200_seq2seq_epoch60
#SBATCH --output=aus_decode_emb200_seq2seq_epoch60
#SBATCH --qos=gpu-short
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --time=1:00:00
#SBATCH --mem=16g

SITENAME=askubuntu_unix_superuser

CQ_DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME

python $SCRIPT_DIR/decode.py		--test_context $CQ_DATA_DIR/test_context.txt \
									--test_ques $CQ_DATA_DIR/test_question.txt \
									--test_ans $CQ_DATA_DIR/test_answer.txt \
									--test_ids $CQ_DATA_DIR/test_ids \
									--test_pred_ques $CQ_DATA_DIR/test_pred_question.txt \
									--q_encoder_params $CQ_DATA_DIR/q_encoder_params.epoch60 \
									--q_decoder_params $CQ_DATA_DIR/q_decoder_params.epoch60 \
									--word_embeddings $EMB_DIR/word_embeddings.p \
									--vocab $EMB_DIR/vocab.p \
									--model seq2seq.epoch60 \
									--max_post_len 100 \
									--max_ques_len 20 \
									--max_ans_len 20 \
									--batch_size 256 \
									--n_epochs 40 \
									--beam True \


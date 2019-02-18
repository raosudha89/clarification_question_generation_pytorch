#!/bin/bash

#SBATCH --job-name=create_crowdflower_HK_beam_batch4
#SBATCH --output=create_crowdflower_HK_beam_batch4
#SBATCH --qos=batch
#SBATCH --mem=32g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
CORPORA_DIR=/fs/clip-corpora/amazon_qa
DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
CROWDFLOWER_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_crowdflower_data.py  --qa_data_fname $CORPORA_DIR/qa_${SITENAME}.json.gz \
                                                --metadata_fname $CORPORA_DIR/meta_${SITENAME}.json.gz \
                                                --batch1_csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_diverse_beam_epoch8.batch1.csv \
                                                --batch2_csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_beam_epoch8.batch2.csv \
                                                --batch3_csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_beam_epoch8.batch3.csv \
                                                --csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_beam_epoch8.batch4.csv \
                                                --lucene_model_name lucene \
                                                --lucene_model_fname $DATA_DIR/blind_test_pred_question.lucene.txt \
                                                --seq2seq_model_name seq2seq.beam \
                                                --seq2seq_model_fname $DATA_DIR/blind_test_pred_question.txt.seq2seq.len_norm.beam0 \
                                                --rl_model_name rl.beam \
                                                --rl_model_fname $DATA_DIR/blind_test_pred_question.txt.RL_selfcritic.epoch8.len_norm.beam0 \
                                                --gan_model_name gan.beam \
                                                --gan_model_fname $DATA_DIR/blind_test_pred_question.txt.GAN_selfcritic_pred_ans_3perid.epoch8.len_norm.beam0 \

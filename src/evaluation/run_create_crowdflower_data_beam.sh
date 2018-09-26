#!/bin/bash

#SBATCH --job-name=create_crowdflower_HK_beam
#SBATCH --output=create_crowdflower_HK_beam
#SBATCH --qos=batch
#SBATCH --mem=32g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
CORPORA_DIR=/fs/clip-corpora/amazon_qa
DATA_DIR=/fs/clip-scratch/raosudha/clarification_question_generation/joint_learning/$SITENAME
CROWDFLOWER_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_crowdflower_data_beam.py  --previous_csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_diverse_beam_epoch8.batch1.csv \
                                                    --output_csv_file $CROWDFLOWER_DIR/crowdflower_rl_beam.batch1.csv \
                                                    --seq2seq_model_name rl.beam \
                                                    --seq2seq_model_fname $DATA_DIR/blind_test_pred_question.txt.RL_selfcritic.epoch8.len_norm.beam0 \
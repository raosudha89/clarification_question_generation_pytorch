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

python $SCRIPT_DIR/create_crowdflower_data_beam.py  --previous_csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_diverse_beam_seq2seq_beam_gan_beam_rl_beam_epoch8.batch1.aggregate.csv \
                                                    --output_csv_file $CROWDFLOWER_DIR/crowdflower_lucene_seq2seq_rl_gan_beam.batch1.csv \
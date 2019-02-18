#!/bin/bash

#SBATCH --job-name=create_crowdflower_HK_beam_spec_single_sent
#SBATCH --output=create_crowdflower_HK_beam_spec_single_sent
#SBATCH --qos=batch
#SBATCH --mem=32g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
CORPORA_DIR=/fs/clip-corpora/amazon_qa
DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
CROWDFLOWER_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_crowdflower_data_specificity.py	--qa_data_fname $CORPORA_DIR/qa_${SITENAME}.json.gz \
                                                			--metadata_fname $CORPORA_DIR/meta_${SITENAME}.json.gz \
                                                			--csv_file $CROWDFLOWER_DIR/crowdflower_seq2seq_epoch100_seq2seq_specific_seq2seq_generic_p100_q30_style_emb_single_sent.epoch100.csv \
															--lucene_model_name lucene \
															--lucene_model_fname $DATA_DIR/blind_test_pred_question.lucene.txt \
															--seq2seq_model_name seq2seq \
															--seq2seq_model_fname $DATA_DIR/blind_test_pred_ques.txt.seq2seq.epoch100.beam0 \
                                                			--seq2seq_specific_model_name seq2seq.specific \
															--seq2seq_specific_model_fname $DATA_DIR/blind_test_pred_ques.txt.seq2seq_tobeginning_tospecific_p100_q30_style_emb.epoch100.beam0 \
                                                			--seq2seq_generic_model_name seq2seq.generic \
															--seq2seq_generic_model_fname $DATA_DIR/blind_test_pred_ques.txt.seq2seq_tobeginning_togeneric_p100_q30_style_emb.epoch100.beam0 \

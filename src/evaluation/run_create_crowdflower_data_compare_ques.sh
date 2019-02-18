#!/bin/bash

#SBATCH --job-name=create_crowdflower_HK_compare_batchD
#SBATCH --output=create_crowdflower_HK_compare_batchD
#SBATCH --qos=batch
#SBATCH --mem=32g
#SBATCH --time=4:00:00

SITENAME=Home_and_Kitchen
CORPORA_DIR=/fs/clip-corpora/amazon_qa
DATA_DIR=/fs/clip-amr/clarification_question_generation_pytorch/$SITENAME
CROWDFLOWER_DIR=/fs/clip-amr/clarification_question_generation_pytorch/evaluation/$SITENAME
SCRIPT_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python $SCRIPT_DIR/create_crowdflower_data_compare_ques.py  --qa_data_fname $CORPORA_DIR/qa_${SITENAME}.json.gz \
                                                			--metadata_fname $CORPORA_DIR/meta_${SITENAME}.json.gz \
                                                			--csv_file $CROWDFLOWER_DIR/crowdflower_compare_ques_batchD_100.csv \
                                                			--train_asins $DATA_DIR/train_asin.txt \
                                                			--previous_csv_file_v1 $CROWDFLOWER_DIR/crowdflower_compare_ques_batchA_100.csv \
                                                			--previous_csv_file_v2 $CROWDFLOWER_DIR/crowdflower_compare_ques_allpairs.csv \
                                                			--previous_csv_file_v3 $CROWDFLOWER_DIR/crowdflower_compare_ques_batchB_100.csv \
                                                			--previous_csv_file_v4 $CROWDFLOWER_DIR/crowdflower_compare_ques_batchC_100.csv \

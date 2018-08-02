#!/bin/bash

#SBATCH --job-name=read_amazon_Electronics
#SBATCH --output=read_amazon_Electronics
#SBATCH --qos=batch
#SBATCH --mem=36g
#SBATCH --time=24:00:00

#SITENAME=Automotive
SITENAME=Electronics
#SITENAME=Home_and_Kitchen

export PATH="/fs/clip-amr/anaconda2/bin:$PATH"

python src/read_amazon_data.py --qa_data_fname /fs/clip-corpora/amazon_qa/qa_${SITENAME}.json.gz --metadata_fname /fs/clip-corpora/amazon_qa/meta_${SITENAME}.json.gz
#python src/read_amazon_data.py --qa_data_fname /fs/clip-corpora/amazon_qa/qa_${SITENAME}.json.gz --metadata_fname /fs/clip-corpora/amazon_qa/metadata.json.gz

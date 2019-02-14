#!/bin/bash

#SITENAME=askubuntu_unix_superuser
SITENAME=Home_and_Kitchen

SCRIPTS_DIR=/fs/clip-amr/clarification_question_generation_pytorch/src/embedding_generation
EMB_DIR=/fs/clip-amr/clarification_question_generation_pytorch/embeddings/$SITENAME/200

python $SCRIPTS_DIR/create_we_vocab.py $EMB_DIR/vectors.txt $EMB_DIR/word_embeddings.p $EMB_DIR/vocab.p


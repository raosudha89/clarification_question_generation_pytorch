export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

NGRAMS_SCRIPT=/fs/clip-amr/clarification_question_generation_pytorch/src/evaluation/all_ngrams.pl

count_uniq_trigrams=$( cat $1 | $NGRAMS_SCRIPT 3 | sort | uniq -c | sort -gr | wc -l )
count_all_trigrams=$( cat $1 | $NGRAMS_SCRIPT 3 | sort | sort -gr | wc -l )
echo "Trigram diversity"
echo "scale=4; $count_uniq_trigrams / $count_all_trigrams" | bc

count_uniq_bigrams=$( cat $1 | $NGRAMS_SCRIPT 2 | sort | uniq -c | sort -gr | wc -l )
count_all_bigrams=$( cat $1 | $NGRAMS_SCRIPT 2 | sort | sort -gr | wc -l )
echo "Bigram diversity"
echo "scale=4; $count_uniq_bigrams / $count_all_bigrams" | bc

count_uniq_unigrams=$( cat $1 | $NGRAMS_SCRIPT 1 | sort | uniq -c | sort -gr | wc -l )
count_all_unigrams=$( cat $1 | $NGRAMS_SCRIPT 1 | sort | sort -gr | wc -l )
echo "Unigram diversity"
echo "scale=4; $count_uniq_unigrams / $count_all_unigrams" | bc



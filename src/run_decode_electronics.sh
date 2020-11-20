python src/decode.py        --test_context baseline_data/valid_context.txt \
									--test_ques baseline_data/valid_question.txt \
									--test_ans baseline_data/valid_answer.txt \
									--test_ids baseline_data/valid_asin.txt \
									--test_pred_ques baseline_data/valid_predicted_question.txt \
									--q_encoder_params baseline_data/seq2seq-pretrain-ques-v3/q_encoder_params.epoch49 \
									--q_decoder_params baseline_data/seq2seq-pretrain-ques-v3/q_decoder_params.epoch49 \
									--word_embeddings embeddings/amazon_200d_embeddings.p \
									--vocab embeddings/amazon_200d_vocab.p \
									--model seq2seq.epoch49 \
									--max_post_len 100 \
									--max_ques_len 20 \
									--max_ans_len 20 \
									--batch_size 10 \
									--n_epochs 40 \
									--greedy True 
									#--beam True

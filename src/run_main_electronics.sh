python src/main.py  --train_context baseline_data/train_context.txt \
                                    --train_ques baseline_data/train_question.txt \
                                    --train_ans baseline_data/train_answer.txt \
				    --train_ids baseline_data/train_asin.txt \
                                    --tune_context baseline_data/valid_context.txt \
                                    --tune_ques baseline_data/valid_question.txt \
                                    --tune_ans baseline_data/valid_answer.txt \
				    --tune_ids baseline_data/valid_asin.txt \
                                    --test_context baseline_data/test_context.txt \
                                    --test_ques baseline_data/test_question.txt \
                                    --test_ans baseline_data/test_answer.txt \
				    --test_ids baseline_data/test_asin.txt \
                                    --q_encoder_params_contd baseline_data/seq2seq-pretrain-ques-v3/q_encoder_params \
                                    --q_decoder_params_contd baseline_data/seq2seq-pretrain-ques-v3/q_decoder_params \
                                    --a_encoder_params_contd baseline_data/seq2seq-pretrain-ans-v3/a_encoder_params \
                                    --a_decoder_params_contd baseline_data/seq2seq-pretrain-ans-v3/a_decoder_params \
                                    --q_encoder_params $PT_OUTPUT_DIR/q_encoder_params \
                                    --q_decoder_params $PT_OUTPUT_DIR/q_decoder_params \
                                    --a_encoder_params $PT_OUTPUT_DIR/a_encoder_params \
                                    --a_decoder_params $PT_OUTPUT_DIR/a_decoder_params \
                                    --context_params $PT_OUTPUT_DIR/context_params \
                                    --question_params $PT_OUTPUT_DIR/question_params \
                                    --answer_params $PT_OUTPUT_DIR/answer_params \
                                    --utility_params $PT_OUTPUT_DIR/utility_params \
                                    --word_embeddings embeddings/amazon_200d_embeddings.p \
                                    --vocab embeddings/amazon_200d_vocab.p \
                                    --n_epochs 100 \
				    --batch_size 128 \
                                    --max_post_len 100 \
				    --max_ques_len 20 \
				    --max_ans_len 20 \
                    --pretrain_ans True \
                    #--pretrain_ques True \				    			    
				    #--pretrain_util True \

USE_CUDA = True

# Configure models
attn_model = 'dot'
hidden_size = 100
n_layers = 1
dropout = 0.5
batch_size = 64
word_emb_size = 200

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 20
epoch = 0.

PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOP_token = '<EOP>'
EOS_token = '<EOS>'
MAX_POST_LEN=300
MAX_QUES_LEN=50
MAX_ANS_LEN=50
MIN_TFIDF=30
clip = 50.0


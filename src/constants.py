USE_CUDA = True

# Configure models
attn_model = 'dot'
#hidden_size = 500
hidden_size = 100
n_layers = 2
dropout = 0.1
#batch_size = 100
batch_size = 256
word_emb_size = 200

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 20
epoch = 0.
plot_every = 20
print_every = 10
evaluate_every = 10

MIN_COUNT = 2
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_POST_LEN=25
MAX_QUES_LEN=25
MIN_TFIDF=30
clip = 50.0


USE_CUDA = True

# Configure models
HIDDEN_SIZE = 100
DROPOUT = 0.5

# Configure training/optimization
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0

PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOP_token = '<EOP>'
EOS_token = '<EOS>'
UNK_token = '<unk>'
SPECIFIC_token = '<specific>'
GENERIC_token = '<generic>'

BEAM_SIZE = 5

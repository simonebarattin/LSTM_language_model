'''
    Configuration file.
'''
########################################################################################################################
##                                                                                                                    ##
##                                                    [ DATASET ]                                                     ##
##                                                                                                                    ##
########################################################################################################################

DATASET = {
    'train': 'ptbdataset/ptb.train.txt',
    'valid': 'ptbdataset/ptb.valid.txt',
    'test':  'ptbdataset/ptb.test.txt'
}

########################################################################################################################
##                                                                                                                    ##
##                                                    [ SAVE PATHS ]                                                  ##
##                                                                                                                    ##
########################################################################################################################

BEST_MODEL_PATH = 'models/model_weights'

########################################################################################################################
##                                                                                                                    ##
##                                                    [ PARAMETERS ]                                                  ##
##                                                                                                                    ##
########################################################################################################################

CLIP_GRADIENT = 0.25
EMBEDDING_SIZE = 400 
HIDDEN_SIZE = 1150
NUM_LAYERS = 3

EPOCHS = 100
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-6
PATIENCE = 5
SEQ_LEN_THRESHOLD = 0.8

# AWD specific
SEQ_LEN_AWD_VANILLA = 70
LR_AWD = 30

# Attention LSTM specific
SEQ_LEN_ATT = 5
TEACHER_FORCING_P = 0.5

# Gated CNN specific
SEQ_LEN_CNN = 21
EMBEDDING_SIZE_CNN = 200
KERNEL_SIZE = 2
OUT_CHANNELS = 600
NUM_LAYERS_CNN = 4
BOTTLENECK = 20
LEARNING_RATE_CNN = 10
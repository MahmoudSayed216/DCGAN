import torch
from utils.logger import LOGGER_SINGLETON, LOG



torch.manual_seed(0)

#training_settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_debugger = False
use_logger = True
use_checkpoint_er = True 
log_every = 30
EPOCHS = 100
save_every = 5
augmentation = True

#dirs
OUTPUT_BASE_DIR = "/kaggle/working/outputs"
DATA_DIR = "/kaggle/input/flickrfaceshq-dataset-ffhq"

#model
LRELU_SLOPE = 0.2
bias = False
G_INITIAL_FMAPS_SIZE = 1024
D_INITIAL_FMAPS_SIZE = 64
weights_mean = 0.0
weights_std = 0.02
bn_mean = 1.0
bn_std = 0.02

#optimizer
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999

  
#i/o
MINI_BATCH_SIZE = 128
LATENT_SPACE_SIZE = 100
OUTPUT_SIZE = 64
OUTPUT_CHANNELS = 3



LOG("HELLO FROM CONFIGS")
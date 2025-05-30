from utils.logger import CHECKPOINT
from datetime import datetime as dt
import torch

def compute_n_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params



def save_checkpoint(file_path, epoch_idx, g_loss, d_loss, gen, disc, gen_optim, disc_optim):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    CHECKPOINT(f"{dt.now()} Saving checkpoints to {file_path}")
    checkpoint = {
        'epoch_idx': epoch_idx,
        'g_loss': g_loss, 
        'd_loss': d_loss,
        'generator': gen.state_dict(),
        'discriminator': disc.state_dict(),
        'gen_optim': gen_optim.state_dict(),
        'disc_optim': disc_optim.state_dict()
    }

    torch.save(checkpoint, file_path)
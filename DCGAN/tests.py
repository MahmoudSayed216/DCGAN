


# from Dataset.Dataset import generate_csv_files, FacesDataset



# generate_csv_files("/home/mahmoud-sayed/Desktop/Code/Python/Models From Scratch/DCGAN/images", "/home/mahmoud-sayed/Desktop/Code/Python/Models From Scratch/DCGAN/csv files", (0.8, 0.1, 0.1))


# ds = FacesDataset("/home/mahmoud-sayed/Desktop/Code/Python/Models From Scratch/DCGAN/csv files/train.csv")

from Model import Generator
from Model import Discriminator
import importlib
importlib.reload(Generator)
import torch

g = Generator.Generator(100, 1024, 0.2)

d = Discriminator.Discriminator(64, 0.2)
BATCH_SIZE = 24
_in = torch.randn(BATCH_SIZE, 100)
output = g(_in)
o = d(output)

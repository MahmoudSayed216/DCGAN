from torch.utils.data import  Dataset
import torch
import pandas as pd
import os
from PIL import Image
from random import shuffle
from torchvision.transforms import transforms as T

def generate_csv_files(data_path, output_path, ratios=(0.8, 0.1, 0.1)) -> list[str]:
    train_csv_path = f"{output_path}/train.csv"
    validation_csv_path = f"{output_path}/val.csv"
    test_csv_path = f"{output_path}/test.csv"

    if os.path.exists(train_csv_path):
        print("CSV FILES ALREADY EXIST")
        return (train_csv_path, validation_csv_path, test_csv_path)

    all_images = os.listdir(data_path)

    n_train = int(ratios[0]*len(all_images))
    n_val = int(ratios[1]*len(all_images))

    shuffle(all_images)

    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train + n_val:]

    append_base_path = lambda lst : [f'{data_path}/{file}' for file in lst]

    train_images = append_base_path(train_images)
    val_images = append_base_path(val_images)
    test_images = append_base_path(test_images)

    train_df = pd.DataFrame({'image_path':train_images})
    val_df = pd.DataFrame({'image_path':val_images})
    test_df = pd.DataFrame({'image_path':test_images})

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(validation_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    return (train_csv_path, validation_csv_path, test_csv_path)


class FacesDataset(Dataset):
    def __init__(self, csv_path, transforms = None):
        self.transforms = transforms
        csv_file = pd.read_csv(csv_path)
        self.images = csv_file['image_path'].to_list()
        if self.transforms is None:
            self.transforms = T.Compose([
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        current_image = self.images[index]
        current_image = Image.open(current_image).convert("RGB")
        current_image = self.transforms(current_image)
             
        current_image = current_image/255.0
        return current_image

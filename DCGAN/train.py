from configs import * 
from Dataset.Dataset import FacesDataset, generate_csv_files
from Model.Generator import Generator
from Model.Discriminator import Discriminator
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch
import torch.nn as nn
from writer import Writer
from utils.logger import LOGGER_SINGLETON, LOG, DEBUG, CHECKPOINT
from utils.utils import compute_n_params, save_checkpoint
import os
from datetime import datetime

def save_generated_images(tensor, save_dir, prefix="img"):
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Rescale from [-1, 1] → [0, 1]
    rescaled = (tensor + 1) / 2.0
    rescaled = torch.clamp(rescaled, 0, 1)

    # Step 2: Save each image
    for idx, img in enumerate(rescaled):
        save_path = os.path.join(save_dir, f"{prefix}_{idx:04d}.png")
        vutils.save_image(img, save_path)

def get_loaders(data_dir, output_base_dir):
    train_csv, val_csv, test_csv = generate_csv_files(data_dir, output_base_dir, (0.8, 0.1, 0.1))

    if augmentation:
        transforms = T.Compose([
            T.Resize((64, 64)),
            T.ColorJitter([0.9,0.9]),
            T.RandomGrayscale(p = 0.3),
            T.RandomAffine((-30,30)),
            T.RandomPerspective(),
            T.GaussianBlur(3),
            T.RandomHorizontalFlip(p = 0.2),
            T.RandomVerticalFlip(p = 0.2),
            # Important parts, above can be ignored
            T.Resize(int(OUTPUT_SIZE * 1.1)),
            T.CenterCrop(OUTPUT_SIZE),
            T.ToTensor()        
        ])    
    
    transforms = T.Compose([
        T.Resize(int(OUTPUT_SIZE * 1.1)),
        T.CenterCrop(OUTPUT_SIZE),
        T.Resize((64, 64)),
        T.ToTensor()        
    ])
    

    train_DS = FacesDataset(train_csv, transforms=transforms)
    val_DS = FacesDataset(val_csv)
    test_DS = FacesDataset(test_csv)

    train_loader = DataLoader(train_DS, batch_size=MINI_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_DS, batch_size=MINI_BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_DS, batch_size=MINI_BATCH_SIZE, shuffle=True)

    return train_loader, val_loader

def validate_model(gen, disc, criterion, val_loader):
    gen.eval()
    disc.eval()
    val_loss_g = 0.0
    val_loss_d = 0.0
    with torch.no_grad():
        for (j, real_imgs)  in enumerate(val_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = torch.ones(batch_size, 1, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device)

            # discriminator
            out_real = disc(real_imgs)
            loss_real = criterion(out_real, real_labels)
            noise = torch.randn(batch_size, LATENT_SPACE_SIZE, device=device)
            fake_imgs = gen(noise)
            out_fake = disc(fake_imgs)
            loss_fake = criterion(out_fake, fake_labels)
            val_loss_d += (loss_real + loss_fake).item()

            # generator
            out_fake_for_g = disc(fake_imgs)
            val_loss_g += criterion(out_fake_for_g, real_labels).item()

        val_loss_d /= len(val_loader)
        val_loss_g /= len(val_loader)
    return val_loss_d, val_loss_g

def train(train_loader, val_loader, output_path):
    
    generator  = Generator(LATENT_SPACE_SIZE, G_INITIAL_FMAPS_SIZE, LRELU_SLOPE).to(device)
    discriminator = Discriminator(D_INITIAL_FMAPS_SIZE, LRELU_SLOPE).to(device)

    gen_n_params = compute_n_params(generator)
    dis_n_params = compute_n_params(discriminator)
    total_n_params = gen_n_params + dis_n_params
    model_size = total_n_params*(4.0/(1024.0*1024.0))
    LOG("TOTAL PARAMS: ", total_n_params)
    LOG("MODEL SIZE", model_size)


    generator._initialize_weights(weights_mean, weights_std, bn_mean, bn_std)
    discriminator._initialize_weights(weights_mean, weights_std, bn_mean, bn_std)

    criterion = nn.BCELoss()

    G_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    D_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    periodic_sample =  torch.randn(MINI_BATCH_SIZE, LATENT_SPACE_SIZE, device=device)

    writer = Writer(output_path)

    highest_D_loss = -1*float('inf')

    for epoch in range(1, EPOCHS):
        generator.train()
        discriminator.train()
        G_running_loss = 0.0
        D_running_loss = 0.0
        for j, real_images in enumerate(train_loader):
            
            discriminator.zero_grad()
            batch_size = real_images.size(0) 
            real_images = real_images.to(device)
            real_labels = torch.ones(batch_size, 1, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device)

            real_disc_output = discriminator(real_images)
            real_loss = criterion(real_disc_output, real_labels)

            #generating fake images
            noise = torch.randn(batch_size, LATENT_SPACE_SIZE, device=device)
            fake_images = generator(noise)
            fake_disc_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_disc_output, fake_labels)

            D_loss = real_loss + fake_loss

            D_loss.backward()
            D_optim.step()

            
            generator.zero_grad()
            out_fake_for_g = discriminator(fake_images)
            G_loss = criterion(out_fake_for_g, real_labels)
            G_loss.backward()
            G_optim.step()
            
            G_running_loss += G_loss.item()
            D_running_loss += D_loss.item()
            if j%60 == 0:
                LOG("ITERATION G LOSS: ",G_loss.item())
                LOG("ITERATION D LOSS: ",D_loss.item())


        G_avg_train_loss = G_running_loss / len(train_loader)
        D_avg_train_loss = D_running_loss / len(train_loader)
        writer.add_scaler("G_AVG_T_LOSS", epoch, G_avg_train_loss)
        writer.add_scaler("D_AVG_T_LOSS", epoch, D_avg_train_loss)
        LOG(f"EPOCH: [{epoch}/{EPOCHS}] | D Train Loss: {D_avg_train_loss:.4f} | G Train Loss: {G_avg_train_loss:.4f}")

        D_avg_val_loss, G_avg_val_loss = validate_model(generator, discriminator, criterion, val_loader)
        LOG(f"EPOCH: [{epoch}/{EPOCHS}] | D Val Loss: {D_avg_val_loss:.4f} | G Val Loss: {G_avg_val_loss:.4f}")
        writer.add_scaler("G_AVG_V_LOSS", epoch, G_avg_val_loss)
        writer.add_scaler("D_AVG_V_LOSS", epoch, D_avg_val_loss)
        
        if D_avg_val_loss > highest_D_loss:
            weights_path = os.path.join(output_path, "weights", "best.pth")
            save_checkpoint(weights_path, epoch, G_avg_val_loss, D_avg_val_loss, generator, discriminator, G_optim, D_optim)
        if epoch ==1 or (epoch%10 == 0):
            generated_images = generator(periodic_sample)
            samples_path = os.path.join(output_path, "samples", f"epoch {epoch}")
            save_generated_images(generated_images, samples_path)

        weights_path = os.path.join(output_path, "weights", "last.pth")
        save_checkpoint(weights_path, epoch, G_avg_val_loss, D_avg_val_loss, generator, discriminator, G_optim, D_optim)

    return generator, discriminator

def initiate_training_environment(path: str):
    if not os.path.exists(path):
        os.mkdir(os.path.join(path))
    new_path = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, "weights"))
    os.mkdir(os.path.join(new_path, "samples"))

    return new_path


def main():
    
    LOGGER_SINGLETON.l_active = use_logger
    LOGGER_SINGLETON.d_active = use_debugger
    LOGGER_SINGLETON.c_active = use_checkpoint_er
    output_path = initiate_training_environment(OUTPUT_BASE_DIR)
    tr_loader, v_loader= get_loaders(DATA_DIR, OUTPUT_BASE_DIR)

    train(tr_loader, v_loader, output_path)




if __name__ == "__main__":
    main()

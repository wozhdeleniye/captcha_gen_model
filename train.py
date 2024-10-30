import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset

import const
from model import Generator, Discriminator


def train_GAN(generator, discriminator, dataloader, epochs, latent_dim, optimizer_G, optimizer_D, criterion,
              save_interval=100, model_save_path='./models'):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            batch_size = real_images.size(0)

            z = torch.randn(batch_size, latent_dim)
            fake_labels = torch.randint(0, const.num_symbols, (batch_size, const.captcha_length))
            # print(fake_labels)
            # fake_labels = encode_text(generate_text())

            fake_images = generator(fake_labels, z)
            print(real_images[0].size())
            # print(fake_images[0].size())
            # print(fake_images[0])

            real_validity = discriminator(real_images, labels)
            fake_validity = discriminator(fake_images.detach(), fake_labels)

            real_loss = criterion(real_validity, torch.ones(batch_size, 1))
            fake_loss = criterion(fake_validity, torch.zeros(batch_size, 1))
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            fake_validity = discriminator(fake_images, fake_labels)
            g_loss = criterion(fake_validity, torch.ones(batch_size, 1))

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if epoch % save_interval == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), os.path.join(model_save_path, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(model_save_path, f'discriminator_epoch_{epoch}.pth'))
            print(f'Модель сохранена на эпохе {epoch}')


def train():
    generator = Generator(embedding_dim=const.embedding_dim, latent_dim=const.latent_dim,
                          captcha_length=const.captcha_length,
                          image_size=const.image_size)
    print("Создан генератор.")

    discriminator = Discriminator(image_size=const.image_size, captcha_length=const.captcha_length,
                                  embedding_dim=const.embedding_dim)
    print("Создан дискриминатор.")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} доступен.")
    else:
        print("GPU не доступен. Тренировка будет происходить на CPU.")

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    print("Созданы оптимитазторы и функция потерь.")

    dataloader = load_dataset()
    print("Датасет загружен.")

    epochs = 1000
    train_GAN(generator, discriminator, dataloader, epochs, const.latent_dim, optimizer_G, optimizer_D, criterion,
              save_interval=const.save_interval, model_save_path=const.models_save_path)

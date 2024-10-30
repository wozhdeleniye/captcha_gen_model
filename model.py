import torch
import torch.nn as nn

import const


class Generator(nn.Module):
    def __init__(self, embedding_dim, latent_dim, captcha_length, image_size):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(const.num_symbols, embedding_dim)

        input_size = embedding_dim * captcha_length + latent_dim
        self.fc = nn.Linear(input_size, 128 * 16 * 32)

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.image_size = image_size

    def forward(self, input_symbols, z):
        embedded = self.embedding(input_symbols).view(input_symbols.size(0), -1)

        gen_input = torch.cat((embedded, z), dim=1)

        x = self.fc(gen_input).view(-1, 128, 16, 32)

        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, image_size, captcha_length, embedding_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(const.num_symbols, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, image, input_symbols):
        x = self.conv(image)
        return x

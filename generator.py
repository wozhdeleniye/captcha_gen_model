import torch
import matplotlib.pyplot as plt

from model import Generator
import const

if __name__ == '__main__':
    model = Generator(const.embedding_dim, const.latent_dim, const.captcha_length, const.image_size)
    model.load_state_dict(torch.load(f'{const.models_save_path}/generator_epoch_200.pth'))

    model.eval()
    input_symbols = torch.randint(0, 6, (const.batch_size, const.captcha_length))
    z = torch.randn(const.batch_size, const.latent_dim)

    with torch.no_grad():
        output_image = model(input_symbols, z)

    output_image_np = output_image[0].permute(1, 2, 0).cpu().numpy()

    plt.imshow((output_image_np + 1) / 2)
    plt.show()

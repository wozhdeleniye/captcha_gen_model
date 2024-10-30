import os

from PIL import Image
from captcha.image import ImageCaptcha
from io import BytesIO
import glob
import random
import string

import const


def generate_captcha(text: str) -> Image:
    fonts = glob.glob('fonts/*')
    captcha: ImageCaptcha = ImageCaptcha(width=const.image_size['width'],
                                         height=const.image_size['height'],
                                         fonts=fonts,
                                         font_sizes=(40, 70, 100))

    data: BytesIO = captcha.generate(text)
    image: Image = Image.open(data)
    return image


def generate_text():
    characters = string.ascii_letters + 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'

    captcha_length = random.randint(4, 6)

    captcha = ''.join(random.choice(characters) for _ in range(captcha_length))

    return captcha


def save_image_with_index(image: Image.Image, directory: str, filename: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    base_filename, ext = os.path.splitext(filename)

    file_path = os.path.join(directory, filename)
    index = 1

    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{base_filename}_{index}{ext}")
        index += 1

    image.save(file_path)
    print(f"Изображение сохранено как: {file_path}")


if __name__ == '__main__':
    for i in range(0, 5000):
        image_text = generate_text()
        save_image_with_index(generate_captcha(image_text), const.dataset_dir, f"{image_text}.png")

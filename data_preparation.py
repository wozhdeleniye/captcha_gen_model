import re
import string
import numpy as np
from PIL import Image
import os

from const import dataset_dir




def load_dataset_path(dataset_dir):
    images_paths = []
    texts = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".png"):
            images_paths.append(os.path.join(dataset_dir, filename))

            text = re.split('[._]', filename)[0]
            texts.append(text)
            print(f'{filename} loaded')

    return np.array(images_paths), texts


def encode_texts(texts, max_len=6):
    chars = sorted(list(string.ascii_letters + 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}

    encoded_texts = []
    for text in texts:
        encoded_text = [char_to_idx.get(char, 0) for char in text]
        encoded_text += [0] * (max_len - len(encoded_text))
        encoded_texts.append(encoded_text)

    return np.array(encoded_texts), char_to_idx


def encode_text(text, max_len=6):
    chars = sorted(list(string.ascii_letters + 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    encoded_text = [char_to_idx.get(char, 0) for char in text]
    encoded_text += [0] * (max_len - len(encoded_text))
    return encoded_text


def make_dataset():
    images_path, texts = load_dataset_path(dataset_dir)
    encoded_texts, char_to_idx = encode_texts(texts)
    return images_path, encoded_texts

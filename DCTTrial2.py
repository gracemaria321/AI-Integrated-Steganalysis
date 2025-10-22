# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 13:31:39 2025

@author: Grace Maria IIT
"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fftpack import dct, idct

# DCT block functions
def blockwise_dct(img_array, block_size=8):
    h, w = img_array.shape
    dct_blocks = np.zeros_like(img_array, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_array[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_blocks[i:i+block_size, j:j+block_size] = dct_block
    return dct_blocks
def blockwise_idct(dct_array, block_size=8):
    h, w = dct_array.shape
    img_array = np.zeros_like(dct_array, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_array[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
                img_array[i:i+block_size, j:j+block_size] = idct_block
    return np.clip(img_array, 0, 255).astype(np.uint8)

# Encode message
def encode_dct(image_path, message, output_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape

    message += chr(0)
    msg_binary = ''.join([format(ord(c), '08b') for c in message])
    msg_index = 0

    dct_array = blockwise_dct(img_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if msg_index >= len(msg_binary):
                break
            block = dct_array[i:i+8, j:j+8]
            coeff = int(block[4, 3])
            block[4, 3] = (coeff & ~1) | int(msg_binary[msg_index])
            dct_array[i:i+8, j:j+8] = block
            msg_index += 1
        if msg_index >= len(msg_binary):
            break

    encoded_array = blockwise_idct(dct_array)
    Image.fromarray(encoded_array).save(output_path)
    return output_path

# Decode message
def decode_dct(image_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape
    dct_array = blockwise_dct(img_array)
    msg_binary = ''
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_array[i:i+8, j:j+8]
            msg_binary += str(int(block[4, 3]) & 1)

    chars = [chr(int(msg_binary[i:i+8], 2)) for i in range(0, len(msg_binary), 8)]
    return ''.join(chars).split(chr(0))[0]

# Evaluation
def evaluate(original_path, stego_path, original_message):
    original = np.array(Image.open(original_path).convert("L"))
    stego = np.array(Image.open(stego_path).convert("L"))
    decoded_message = decode_dct(stego_path)

    ssim_val = ssim(original, stego)
    psnr_val = psnr(original, stego)
    accuracy = sum(o == d for o, d in zip(original_message, decoded_message)) / len(original_message) * 100

    print("SSIM:", round(ssim_val, 4))
    print("PSNR:", round(psnr_val, 2), "dB")
    print("Decoded Message:", decoded_message)
    print("Detection Accuracy:", round(accuracy, 2), "%")

# Example usage 1
original_image = "images/picture.jpg"
stego_image = "outputImages/picture_dct_encoded.png"
message = "Hello Grace! Welcome to IIT. Steganography is fun. AI enhances security. Python scripting rocks!"

encode_dct(original_image, message, stego_image)
evaluate(original_image, stego_image, message)

# Example usage 2
original_image = "images/boots.png"
stego_image = "outputImages/boots_dct_encoded.png"
message = "Hello Grace! Welcome to IIT. Steganography is fun. AI enhances security. Python scripting rocks!"

encode_dct(original_image, message, stego_image)
evaluate(original_image, stego_image, message)
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 13:10:19 2025

@author: Grace Maria IIT
"""

import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fftpack import dct, idct
import kagglehub

import warnings
warnings.filterwarnings("ignore", message=".*kagglehub.*")


# Helper functions for DCT
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

# DCT-based encoding
def encode_dct(image_path, message, output_path=None):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape

    message += chr(0)  # Delimiter
    msg_binary = ''.join([format(ord(char), '08b') for char in message])
    msg_index = 0

    dct_array = blockwise_dct(img_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if msg_index >= len(msg_binary):
                break
            block = dct_array[i:i+8, j:j+8]
            if block.shape == (8, 8):
                coeff = int(block[4, 3])
                coeff = (coeff & ~1) | int(msg_binary[msg_index])
                block[4, 3] = coeff
                dct_array[i:i+8, j:j+8] = block
                msg_index += 1
        if msg_index >= len(msg_binary):
            break

    encoded_array = blockwise_idct(dct_array)
    encoded_img = Image.fromarray(encoded_array)
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_dct_encoded.png"
    encoded_img.save(output_path)
    return output_path

# DCT-based decoding
def decode_dct(image_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape

    dct_array = blockwise_dct(img_array)
    msg_binary = ''
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_array[i:i+8, j:j+8]
            if block.shape == (8, 8):
                coeff = int(block[4, 3])
                msg_binary += str(coeff & 1)

    chars = [chr(int(msg_binary[i:i+8], 2)) for i in range(0, len(msg_binary), 8)]
    message = ''.join(chars)
    return message.split(chr(0))[0]

# Evaluation metrics
def compute_ssim(original_path, stego_path):
    original = np.array(Image.open(original_path).convert("L"))
    stego = np.array(Image.open(stego_path).convert("L"))
    return ssim(original, stego)

def compute_psnr(original_path, stego_path):
    original = np.array(Image.open(original_path).convert("L"))
    stego = np.array(Image.open(stego_path).convert("L"))
    return psnr(original, stego)

def detection_accuracy(original_message, decoded_message):
    correct = sum(o == d for o, d in zip(original_message, decoded_message))
    return (correct / len(original_message)) * 100 if original_message else 0

# Dataset processing
def process_dct_dataset(image_folder, messages):
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))])
    log = []

    for img_file, message in zip(image_files, messages):
        img_path = os.path.join(image_folder, img_file)
        base, ext = os.path.splitext(img_file)
        stego_path = os.path.join(image_folder, f"{base}_dct_encoded.png")

        encode_dct(img_path, message, stego_path)
        ssim_val = compute_ssim(img_path, stego_path)
        psnr_val = compute_psnr(img_path, stego_path)
        decoded_msg = decode_dct(stego_path)
        accuracy = detection_accuracy(message, decoded_msg)

        log.append({
            "image": img_file,
            "stego_image": os.path.basename(stego_path),
            "message": message,
            "decoded": decoded_msg,
            "SSIM": round(ssim_val, 4),
            "PSNR": round(psnr_val, 2),
            "Accuracy": round(accuracy, 2)
        })

    for entry in log:
        print(f"\nImage: {entry['image']}")
        print(f"Stego Image: {entry['stego_image']}")
        print(f"Original Message: {entry['message']}")
        print(f"Decoded Message: {entry['decoded']}")
        print(f"SSIM: {entry['SSIM']}")
        print(f"PSNR: {entry['PSNR']} dB")
        print(f"Detection Accuracy: {entry['Accuracy']}%")

dataset_path = kagglehub.dataset_download("saadghojaria/sneakers-image-dataset-pinterest")
messages_list = [
    "Hello Grace!",
    "Welcome to IIT.",
    "Steganography is fun.",
    "AI enhances security.",
    "Python scripting rocks!"
]
process_dct_dataset(dataset_path, messages_list)

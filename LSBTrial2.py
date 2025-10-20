# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 21:46:53 2025

@author: Grace Maria IIT
"""

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import sobel
import kagglehub
import os

# Function to compute SSIM
def compute_ssim(original_path, stego_path):
    original = Image.open(original_path).convert("RGB")
    stego = Image.open(stego_path).convert("RGB")
    original_np = np.array(original)
    stego_np = np.array(stego)
    return ssim(original_np, stego_np, channel_axis=-1, win_size=3)

# Function to compute PSNR
def compute_psnr(original_path, stego_path):
    original = Image.open(original_path).convert("RGB")
    stego = Image.open(stego_path).convert("RGB")
    original_np = np.array(original)
    stego_np = np.array(stego)
    return psnr(original_np, stego_np)

# Function to compute detection accuracy
def detection_accuracy(original_message, decoded_message):
    correct = sum(o == d for o, d in zip(original_message, decoded_message))
    return (correct / len(original_message)) * 100 if original_message else 0

# Function to encode message using LSB with edge detection
def encode_lsb(image_path, message, output_path=None):
    img = Image.open(image_path).convert("RGB")
    encoded = img.copy()
    width, height = img.size

    message += chr(0)  # Delimiter
    msg_binary = ''.join([format(ord(char), '08b') for char in message])
    msg_index = 0

    gray = np.array(img.convert("L"))
    edge_map = sobel(gray)
    edge_threshold = np.percentile(edge_map, 75)

    for y in range(height):
        for x in range(width):
            if msg_index < len(msg_binary) and edge_map[y, x] >= edge_threshold:
                r, g, b = img.getpixel((x, y))
                r = (r & ~1) | int(msg_binary[msg_index])
                encoded.putpixel((x, y), (r, g, b))
                msg_index += 1
            if msg_index >= len(msg_binary):
                break
        if msg_index >= len(msg_binary):
            break

    original_format = Image.open(image_path).format
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_encoded.{original_format.lower()}"

    encoded.save(output_path, format=original_format)
    print("Message encoded and saved to", output_path)
    return output_path

# Function to decode message from stego image
def decode_lsb(image_path):
    img = Image.open(image_path)
    width, height = img.size
    msg_binary = ''

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            msg_binary += str(r & 1)

    chars = [chr(int(msg_binary[i:i+8], 2)) for i in range(0, len(msg_binary), 8)]
    message = ''.join(chars)
    return message.split(chr(0))[0]

# Function to process dataset of images with corresponding messages
def process_dataset(image_folder, messages):
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))])
    log = []

    for img_file, message in zip(image_files, messages):
        img_path = os.path.join(image_folder, img_file)
        base, ext = os.path.splitext(img_file)
        stego_path = os.path.join(image_folder, f"{base}_encoded{ext}")

        encode_lsb(img_path, message, stego_path)
        ssim_val = compute_ssim(img_path, stego_path)
        psnr_val = compute_psnr(img_path, stego_path)
        decoded_msg = decode_lsb(stego_path)
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

    # Print log
    for entry in log:
        print(f"\nImage: {entry['image']}")
        print(f"Stego Image: {entry['stego_image']}")
        print(f"Original Message: {entry['message']}")
        print(f"Decoded Message: {entry['decoded']}")
        print(f"SSIM: {entry['SSIM']}")
        print(f"PSNR: {entry['PSNR']} dB")
        print(f"Detection Accuracy: {entry['Accuracy']}%")

# Example usage
# Define your dataset path and messages
dataset_path = kagglehub.dataset_download("saadghojaria/sneakers-image-dataset-pinterest")
messages_list = [
    "Hello Grace!",
    "Welcome to IIT.",
    "Steganography is fun.",
    "AI enhances security.",
    "Python scripting rocks!"
]

# Process the dataset
process_dataset(dataset_path, messages_list)
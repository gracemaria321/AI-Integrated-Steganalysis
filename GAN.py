# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:10:21 2025

@author: Grace Maria IIT

GAN-based Steganography Example
"""

from steganogan import SteganoGAN
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# SSIM and PSNR Functions
def compute_ssim(original_path, stego_path):
    original = np.array(Image.open(original_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))
    value = ssim(original, stego, channel_axis=-1)
    print(f"SSIM: {value:.4f}")

def compute_psnr(original_path, stego_path):
    original = np.array(Image.open(original_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))
    value = psnr(original, stego)
    print(f"PSNR: {value:.2f} dB")

# Detection Accuracy
def detection_accuracy(original_message, decoded_message):
    correct = sum(o == d for o, d in zip(original_message, decoded_message))
    return (correct / len(original_message)) * 100

# Ensure output directory exists
os.makedirs("outputImages", exist_ok=True)

# Load the built-in pretrained GAN model
model = SteganoGAN.load(architecture='dense', cuda=False, verbose=True)

# File paths
original_image = "images/sneakers.png"
stego_image = "outputImages/sneakers_encoded.png"
secret_message = "Hello Grace! Welcome to IIT. Steganography is fun. AI enhances security. Python scripting rocks!"

# Check if original image exists
if not os.path.exists(original_image):
    raise FileNotFoundError(f"Original image not found at: {original_image}")

# Hide the message
model.encode_image(original_image, stego_image, secret_message)
print("Message hidden!")

# Reveal the message
decoded_message = model.decode_image(stego_image)
print("Decoded Message:", decoded_message)

# Evaluate image quality
compute_ssim(original_image, stego_image)
compute_psnr(original_image, stego_image)

# Evaluate detection accuracy
accuracy = detection_accuracy(secret_message, decoded_message)
print(f"Detection Accuracy: {accuracy:.2f}%")
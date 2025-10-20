# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:44:03 2025

@author: Grace Maria IIT
"""

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import sobel
import os

def SSIM(original, stego):    
    # Load images
    original = Image.open("images/cyber.jpg").convert("RGB")
    stego = Image.open("outputImages/cyber_encoded.jpeg").convert("RGB")
    
    # Convert to NumPy arrays
    original_np = np.array(original)
    stego_np = np.array(stego)
    
    # Compute SSIM
    ssim_value = ssim(original_np, stego_np, channel_axis=-1, win_size=3)
    
    # Display results
    print(f"SSIM: {ssim_value:.4f}")

def PSNR(original, stego): 
    # Load images
    original = Image.open("images/cyber.jpg").convert("RGB")
    stego = Image.open("outputImages/cyber_encoded.jpeg").convert("RGB")
    
    # Convert to NumPy arrays
    original_np = np.array(original)
    stego_np = np.array(stego)
           
    # Compute PSNR
    psnr_value = psnr(original_np, stego_np)
    
    # Display results
    print(f"PSNR: {psnr_value:.2f} dB")

def detection_accuracy(original_message, decoded_message):
    correct = sum(o == d for o, d in zip(original_message, decoded_message))
    return (correct / len(original_message)) * 100

def encode_lsb(image_path, message, output_path=None):
    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    encoded = img.copy()
    width, height = img.size

    # Prepare message
    message += chr(0)  # Delimiter
    msg_binary = ''.join([format(ord(char), '08b') for char in message])
    msg_index = 0

    # Convert to grayscale and compute edge map
    gray = np.array(img.convert("L"))
    edge_map = sobel(gray)
    edge_threshold = np.percentile(edge_map, 25)  # Top 75% edge pixels
    
    # Check Message Too Long for Available Pixels
    if len(msg_binary) > np.sum(edge_map >= edge_threshold):
        print("Warning: Not enough edge pixels to embed the full message.")

    # Embed message in high-edge regions
    for y in range(height):
        for x in range(width):
            if msg_index < len(msg_binary) and edge_map[y, x] >= edge_threshold:
                r, g, b = img.getpixel((x, y))
                r = (r & ~1) | int(msg_binary[msg_index])
                encoded.putpixel((x, y), (r, g, b))
                msg_index += 1
        if msg_index >= len(msg_binary):
            break

    # Determine original format
    original_format = Image.open(image_path).format
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join("outputImages", f"{name}_encoded.{original_format.lower()}")
        # Make sure the images/outputImages/ folder exists
        os.makedirs("outputImages", exist_ok=True)
    
    # Save encoded image in original format
    encoded.save(output_path, format=original_format)
    print("Message encoded and saved to", output_path)

def decode_lsb(image_path):
    # Load the stego image
    img = Image.open(image_path)
    width, height = img.size
    msg_binary = ''

    # Extract LSBs from red channel
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            msg_binary += str(r & 1)

    # Convert binary to characters
    chars = [chr(int(msg_binary[i:i+8], 2)) for i in range(0, len(msg_binary), 8)]
    message = ''.join(chars)

    # Stop at delimiter (null character)
    return message.split(chr(0))[0]


encode_lsb("images/cyber.jpg", "Hello Grace Maria!")  # This saves as .jpeg format
SSIM("images/cyber.jpg", "outputImages/cyber_encoded.jpeg")
PSNR("images/cyber.jpg", "outputImages/cyber_encoded.jpeg")
decoded = decode_lsb("outputImages/cyber_encoded.jpeg")
accuracy = detection_accuracy("Hello Grace Maria!", decoded)
print("Decoded Message:", decoded)
print("Detection Accuracy:", accuracy)

"""
Step 1, set the parameter "act_images_folder_absolute_path" to the absolute path of the folder where the original images are stored
Step 2, set the parameter "attack_sample_folder_absolute_path" to the absolute path of the folder where the adversarial samples are stored
"""


import torch
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path to the folder where the actual images
actual_images_folder_absolute_path = "..."

attack_sample_folder_absolute_path = r"..."

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

# Record the total number of original images
image_num = 0

psnr_current = 0

psnr_total = 0

for image_file in image_files:
    print(f"The image number currently being processed is: {image_file}")
    image_num = image_num + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

    attack_sample_absolute_path = os.path.join(attack_sample_folder_absolute_path, image_file)

    image1 = imread(actual_image_absolute_path)
    image2 = imread(attack_sample_absolute_path)

    psnr_current = psnr(image1, image2)
    psnr_total = psnr_total + psnr_current

print(psnr_total / image_num)
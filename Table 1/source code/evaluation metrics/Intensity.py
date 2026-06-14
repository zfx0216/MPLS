"""
Step 1, set the parameter "act_images_folder_absolute_path" to the absolute path of the folder where the original images are stored
Step 2, set the parameter "adv_images_folder_absolute_path" to the absolute path of the folder where the adversarial samples are stored
"""

import torch
import numpy as np
from PIL import Image
import os
import math


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path to the folder where the actual images
actual_images_folder_absolute_path = r"..."

# The absolute path to the folder where the actual images
adv_images_folder_absolute_path = r"..."


image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

difference_mean = 0

image_num = 1

for image_file in image_files:
    print(f"The image number currently being processed is: {image_file}")
    image_num = image_num + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

    adv_image_absolute_path = os.path.join(adv_images_folder_absolute_path, image_file)

    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    # The R, G, B three channel matrix of the actual image
    actual_image_matrix[0] = actual_image[:, :, 0]
    actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
    actual_image_matrix[1] = actual_image[:, :, 1]
    actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
    actual_image_matrix[2] = actual_image[:, :, 2]
    actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

    image = Image.open(adv_image_absolute_path)
    image = image.resize((224, 224))
    adv_image = np.array(image)

    adv_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    # The R, G, B three channel matrix of the actual image
    adv_image_matrix[0] = adv_image[:, :, 0]
    adv_image_matrix[0] = adv_image_matrix[0].astype(np.float64)
    adv_image_matrix[1] = adv_image[:, :, 1]
    adv_image_matrix[1] = adv_image_matrix[1].astype(np.float64)
    adv_image_matrix[2] = adv_image[:, :, 2]
    adv_image_matrix[2] = adv_image_matrix[2].astype(np.float64)

    difference_matrix = np.abs(actual_image_matrix - adv_image_matrix)
    difference_mean = difference_mean + np.sum(difference_matrix) / 3 / 224 / 224

print(difference_mean / image_num)


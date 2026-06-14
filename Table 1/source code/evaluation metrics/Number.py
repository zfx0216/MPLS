"""
Step 1, set the parameter "act_images_folder_absolute_path" to the absolute path of the folder where the original images are stored
Step 2, set the parameter "attack_sample_folder_absolute_path" to the absolute path of the folder where the adversarial samples are stored
"""


import torch
import numpy as np
from PIL import Image
import os


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


def count_differences(actual_image_matrix, attack_sample_matrix):
    diff = np.sum(actual_image_matrix != attack_sample_matrix)
    return diff


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path to the folder where the actual images
actual_images_folder_absolute_path = "..."

attack_sample_folder_absolute_path = r"..."

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

# Record the total number of original images
image_num = 0

different_pixel_num_total = 0

different_pixel_num_current = 0

for image_file in image_files:
    print(f"The image number currently being processed is: {image_file}")
    image_num = image_num + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_matrix[0] = actual_image[:, :, 0]
    actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
    actual_image_matrix[1] = actual_image[:, :, 1]
    actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
    actual_image_matrix[2] = actual_image[:, :, 2]
    actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

    attack_sample_absolute_path = os.path.join(attack_sample_folder_absolute_path, image_file)

    image = Image.open(attack_sample_absolute_path)
    image = image.resize((224, 224))
    attack_sample = np.array(image)

    # The R, G, B three channel matrix of the actual image
    attack_sample_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    attack_sample_matrix[0] = attack_sample[:, :, 0]
    attack_sample_matrix[0] = attack_sample_matrix[0].astype(np.float64)
    attack_sample_matrix[1] = attack_sample[:, :, 1]
    attack_sample_matrix[1] = attack_sample_matrix[1].astype(np.float64)
    attack_sample_matrix[2] = attack_sample[:, :, 2]
    attack_sample_matrix[2] = attack_sample_matrix[2].astype(np.float64)

    different_pixel_num_current = count_differences(actual_image_matrix, attack_sample_matrix)
    print(different_pixel_num_current)

    different_pixel_num_total = different_pixel_num_total + different_pixel_num_current

print(different_pixel_num_total / image_num)

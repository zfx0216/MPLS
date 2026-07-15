"""
This code file is used to generate interference information images between the original image and adversarial samples
Step 1: You need to set the 'read_file_math_mact_image' parameter to the absolute path of the original image
Step 2: You need to set the parameter 'read_file_path_ allsaria_sample' to the absolute path of the adversarial sample
Step 3: You need to set the "save_file_math" parameter to the absolute path for saving the final interference information image
Finally execute the file
"""


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

# Read the actual image to be analyzed
read_file_path_actual_image = "..."
# Open the image file
image = Image.open(read_file_path_actual_image)
# Resize the image to the specified size
image = image.resize((224, 224))
# Convert the image to a NumPy array
image_array = np.array(image)

# Based on the RGB display mode, extract the data of each channel
image_red_channel_actual = image_array[:,:,0]
image_green_channel_actual = image_array[:,:,1]
image_blue_channel_actual = image_array[:,:,2]

image_red_channel_actual = image_red_channel_actual.astype(np.float64)
image_green_channel_actual = image_green_channel_actual.astype(np.float64)
image_blue_channel_actual = image_blue_channel_actual.astype(np.float64)

# Read the Adversarial sample to be analyzed
read_file_path_adversarial_sample = "..."
# Open the image file
image = Image.open(read_file_path_adversarial_sample)
# Resize the image to the specified size
image = image.resize((224, 224))
# Convert the image to a NumPy array
image_array = np.array(image)

# Based on the RGB display mode, extract the data of each channel
image_red_channel_attack = image_array[:,:,0]
image_green_channel_attack = image_array[:,:,1]
image_blue_channel_attack = image_array[:,:,2]

image_red_channel_attack = image_red_channel_attack.astype(np.float64)
image_green_channel_attack = image_green_channel_attack.astype(np.float64)
image_blue_channel_attack = image_blue_channel_attack.astype(np.float64)


diff_red_channel = np.abs(image_red_channel_attack - image_red_channel_actual)

diff_green_channel = np.abs(image_green_channel_attack - image_green_channel_actual)

diff_blue_channel = np.abs(image_blue_channel_attack - image_blue_channel_actual)

for i in range(224):
    for j in range(224):
        if diff_red_channel[i][j] == 0 and diff_green_channel[i][j] == 0 and diff_blue_channel[i][j] == 0:
           diff_red_channel[i][j] = 255 - diff_red_channel[i][j]
           diff_green_channel[i][j] = 255 - diff_green_channel[i][j]
           diff_blue_channel[i][j] = 255 - diff_blue_channel[i][j]

rgb_image = np.stack((diff_red_channel, diff_green_channel, diff_blue_channel), axis=-1)

plt.imshow(rgb_image.astype(np.uint8))
plt.axis('off')

#  to the absolute path for saving the final interference information image
save_file_path = "..."

plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0)  # Save the image without extra whitespace
plt.show()
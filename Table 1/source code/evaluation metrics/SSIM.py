"""
Step 1, set the parameter "folder1_path" to the absolute path of the folder where the original images are stored
Step 2, set the parameter "folder2_path" to the absolute path of the folder where the adversarial samples are stored
"""


import os
from skimage.metrics import structural_similarity as ssim
from skimage import io


def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3)


def calculate_ssim_for_folders(folder1, folder2):

    file_list1 = os.listdir(folder1)
    file_list2 = os.listdir(folder2)

    if len(file_list1) != len(file_list2):
        raise ValueError("The quantity in the two folders i not equal")

    ssim_results = []
    total_ssim = 0

    for file_name1, file_name2 in zip(file_list1, file_list2):

        if file_name1.endswith(('.png', '.jpg', '.jpeg', '.bmp')) and file_name2.endswith(
                ('.png', '.jpg', '.jpeg', '.bmp')):

            img1_path = os.path.join(folder1, file_name1)
            img2_path = os.path.join(folder2, file_name2)

            img1 = io.imread(img1_path)
            img2 = io.imread(img2_path)

            ssim_value = calculate_ssim(img1, img2)
            print(ssim_value)

            ssim_results.append((file_name1, file_name2, ssim_value))
            total_ssim += ssim_value

    average_ssim = total_ssim / len(ssim_results) if ssim_results else 0

    print(f"The average SSIM is:{average_ssim}")

    return ssim_results, average_ssim

# Load image
folder1_path = "..."
folder2_path = r"..."

ssim_results, average_ssim = calculate_ssim_for_folders(folder1_path, folder2_path)
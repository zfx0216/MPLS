import os
from skimage.metrics import structural_similarity as ssim
from skimage import io
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def calculate_ssim(img1, img2):
    try:
        if img1.shape != img2.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(img1.shape, img2.shape))
            img1 = img1[:min_shape[0], :min_shape[1]]
            img2 = img2[:min_shape[0], :min_shape[1]]
            if len(img1.shape) == 3 and len(img2.shape) == 3:
                img1 = img1[:, :, :min_shape[2]]
                img2 = img2[:, :, :min_shape[2]]

        if len(img1.shape) == 3 and len(img2.shape) == 3 and img1.shape[-1] == 3:
            ssim_r = ssim(img1[..., 0], img2[..., 0], win_size=3, data_range=img1.max() - img1.min())
            ssim_g = ssim(img1[..., 1], img2[..., 1], win_size=3, data_range=img1.max() - img1.min())
            ssim_b = ssim(img1[..., 2], img2[..., 2], win_size=3, data_range=img1.max() - img1.min())
            ssim_value = (ssim_r + ssim_g + ssim_b) / 3
        else:
            img1_gray = img1 if len(img1.shape) == 2 else np.mean(img1, axis=-1)
            img2_gray = img2 if len(img2.shape) == 2 else np.mean(img2, axis=-1)
            ssim_value = ssim(img1_gray, img2_gray, win_size=3, data_range=img1_gray.max() - img1_gray.min())

        return ssim_value
    except Exception as e:
        print(f"SSIM calculation error: {e}")
        return 0.0

def calculate_ssim_for_two_folders(folder1, folder2, pair_name="1-2"):
    def get_sorted_image_files(folder):
        image_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        files = [f for f in os.listdir(folder) if f.lower().endswith(image_ext)]
        files.sort(key=lambda x: os.path.splitext(x)[0])
        return files

    file_list1 = get_sorted_image_files(folder1)
    file_list2 = get_sorted_image_files(folder2)

    min_file_count = min(len(file_list1), len(file_list2))
    if min_file_count == 0:
        raise ValueError(f"{pair_name}: No images found in at least one folder!")

    if len(file_list1) != len(file_list2):
        print(f"Warning {pair_name}: Image counts mismatch! Folder1: {len(file_list1)}, Folder2: {len(file_list2)}, comparing first {min_file_count} images only")

    ssim_dict = {}
    total_ssim = 0.0
    max_ssim = 0.0

    print(f"\n========== Calculating SSIM for {pair_name} ==========")
    for i in range(min_file_count):
        file1 = file_list1[i]
        file2 = file_list2[i]

        img1_path = os.path.join(folder1, file1)
        img2_path = os.path.join(folder2, file2)

        try:
            img1 = io.imread(img1_path)
            img2 = io.imread(img2_path)

            ssim_value = calculate_ssim(img1, img2)

            ssim_dict[file1] = ssim_value
            total_ssim += ssim_value

            if ssim_value > max_ssim:
                max_ssim = ssim_value

            print(f"File: {file1} vs {file2} | SSIM: {ssim_value:.4f}")

        except Exception as e:
            print(f"Error processing {file1} vs {file2}: {e}")
            ssim_dict[file1] = 0.0

    average_ssim = total_ssim / min_file_count if min_file_count > 0 else 0.0

    print(f"\n{pair_name} Statistics:")
    print(f"Valid image pairs: {min_file_count}")
    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Max SSIM: {max_ssim:.4f}")

    return ssim_dict, average_ssim, max_ssim

def compare_ssim_results(ssim_dict_12, ssim_dict_13):
    print("\n========== SSIM Comparison Analysis ==========")

    common_files = set(ssim_dict_12.keys()) & set(ssim_dict_13.keys())
    if not common_files:
        print("No common files found for comparison!")
        return 0, []

    higher_count = 0
    lower_count = 0
    equal_count = 0
    higher_files = []

    print(f"Common image pairs: {len(common_files)}")
    print("\nSSIM Comparison (Folder1-3 > Folder1-2):")
    print("-" * 60)
    print(f"{'Filename':<30} {'1-2 SSIM':<10} {'1-3 SSIM':<10} {'Result'}")
    print("-" * 60)

    for file in sorted(common_files):
        ssim_12 = ssim_dict_12[file]
        ssim_13 = ssim_dict_13[file]

        if ssim_13 > ssim_12:
            higher_count += 1
            higher_files.append(file)
            result = "Higher"
        elif ssim_13 < ssim_12:
            lower_count += 1
            result = "Lower"
        else:
            equal_count += 1
            result = "Equal"

        print(f"{file:<30} {ssim_12:.4f}      {ssim_13:.4f}      {result}")

    print("-" * 60)
    print(f"\nSummary:")
    print(f"Folder1-3 SSIM > Folder1-2: {higher_count}")
    print(f"Folder1-3 SSIM < Folder1-2: {lower_count}")
    print(f"Folder1-3 SSIM = Folder1-2: {equal_count}")
    print(f"\nFiles with higher SSIM: {higher_files}")

    return higher_count, higher_files

if __name__ == "__main__":
    folder1_path = "..."
    folder2_path = "..."
    folder3_path = "..."

    ssim_12, avg_12, max_12 = calculate_ssim_for_two_folders(
        folder1_path, folder2_path,
        pair_name="Folder1-Folder2"
    )

    ssim_13, avg_13, max_13 = calculate_ssim_for_two_folders(
        folder1_path, folder3_path,
        pair_name="Folder1-Folder3"
    )

    higher_count, higher_files = compare_ssim_results(ssim_12, ssim_13)

    print("\n========== Final Summary ==========")
    print(f"Folder1-2 Average SSIM: {avg_12:.5f}")
    print(f"Folder1-3 Average SSIM: {avg_13:.5f}")
    print(f"Folder1-3 SSIM > Folder1-2: {higher_count} images")
    print(f"Ratio: {higher_count / len(ssim_12) * 100:.2f}%" if len(ssim_12) > 0 else "No data")

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter



def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

###########################################################################################################
# Import the absolute path of the actual image
actual_image_absolute_path = r"G:\决策边界实验\解释性可视化\原始图像\1.png"
image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
image_array_actual = np.array(image)

# RGB three-channel matrix of the actual image
actual_image_R_channel = image_array_actual[:, :, 0]
actual_image_G_channel = image_array_actual[:, :, 1]
actual_image_B_channel = image_array_actual[:, :, 2]

# Read the pixel weight matrix of the RGB three-channel of the actual image
pixel_weight_matrix_R_path = r"G:\决策边界实验\解释性可视化\原始图像梯度文件\baseline\1\pixel_weight_matrix_R.txt"
pixel_weight_matrix_G_path = r"G:\决策边界实验\解释性可视化\原始图像梯度文件\baseline\1\pixel_weight_matrix_G.txt"
pixel_weight_matrix_B_path = r"G:\决策边界实验\解释性可视化\原始图像梯度文件\baseline\1\pixel_weight_matrix_B.txt"

pixel_weight_matrix_R = read_matrix(pixel_weight_matrix_R_path)
pixel_weight_matrix_G = read_matrix(pixel_weight_matrix_G_path)
pixel_weight_matrix_B = read_matrix(pixel_weight_matrix_B_path)

# Calculate the pixel classification contribution matrix for the RGB three channels
pixel_classification_contribution_matrix_R = ((actual_image_R_channel / 255) - 0.485) / 0.229 * pixel_weight_matrix_R
pixel_classification_contribution_matrix_G = ((actual_image_G_channel / 255) - 0.456) / 0.224 * pixel_weight_matrix_G
pixel_classification_contribution_matrix_B = ((actual_image_B_channel / 255) - 0.406) / 0.225 * pixel_weight_matrix_B

# Calculate the total pixel classification contribution matrix
pixel_classification_contribution_matrix_total = np.abs(pixel_classification_contribution_matrix_R) + np.abs(pixel_classification_contribution_matrix_G) + np.abs(pixel_classification_contribution_matrix_B)

pixel_classification_contribution_matrix_total_plot = pixel_classification_contribution_matrix_total.copy()

smoothed = median_filter(pixel_classification_contribution_matrix_total, size=3)
smoothed = gaussian_filter(smoothed, sigma=20)
pixel_classification_contribution_matrix_total = smoothed.copy()
flattened = pixel_classification_contribution_matrix_total.flatten()
sorted_indices = np.argsort(flattened)
ranks = np.empty_like(sorted_indices)
ranks[sorted_indices] = np.arange(1, len(flattened) + 1)
pixel_classification_contribution_matrix_total = ranks.reshape(pixel_classification_contribution_matrix_total.shape)
pixel_classification_contribution_matrix_total = (pixel_classification_contribution_matrix_total - pixel_classification_contribution_matrix_total.min()) / (pixel_classification_contribution_matrix_total.max() - pixel_classification_contribution_matrix_total.min() + 1e-8)
img = Image.open(actual_image_absolute_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., pixel_classification_contribution_matrix_total, use_rgb=True)
image = Image.fromarray((visualization).astype(np.uint8))
image.save("saliency.png")
img = Image.open(actual_image_absolute_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
input_img = img.astype(dtype=np.float32) / 255.
visualization = show_cam_on_image(input_img, pixel_classification_contribution_matrix_total, use_rgb=True)
overlay_image = Image.fromarray((visualization).astype(np.uint8))
overlay_image.save("saliency.png")
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(visualization)
ax.axis('off')

vmin = pixel_classification_contribution_matrix_total_plot.min()
vmax = pixel_classification_contribution_matrix_total_plot.max()

num_ticks = 4
tick_values = np.linspace(vmin, vmax, num=num_ticks)
tick_labels = [f"{val:.2f}" for val in tick_values]

cbar = plt.colorbar(
    mappable=plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=vmin, vmax=vmax)),
    ax=ax,
    fraction=0.046, pad=0.04
)
cbar.set_ticks(tick_values)
cbar.set_ticklabels(tick_labels)

plt.tight_layout()

plt.savefig('saliency_map.png', bbox_inches='tight')
plt.show()
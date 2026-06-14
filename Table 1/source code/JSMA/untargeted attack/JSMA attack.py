"""
The primary function of this code file is to perform untargeted attacks on a specified convolutional neural network using JSMA based on the L0 norm.
First, set `folder_path` to the absolute path of the folder containing the original images.
Second, set `save_path_for_adversarial_samples` to the absolute path of the folder where the adversarial samples will be saved.
Third, set `json_path` to the absolute path of the index file.
Fourth, set `weights_path` to the absolute path of the weight file.
Fifth, use `from torchvision.models import googlenet` to import the official convolutional neural network, and specify it using `model_current`.
Sixth, control the number of tampered pixels by setting the value of `ratio`.
Finally, execute the code file to generate the adversarial samples.
"""


import os
import torch
from PIL import Image
from torchvision.models import googlenet
from torchvision import transforms
import shutil
import matplotlib.pyplot as plt
import math
import numpy as np



def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def saliency_map(image_path, label, weight_file_path, model_cnn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model = model_cnn(num_classes=1000).to(device)
    model.to(device)
    img = img.to(device)
    model.load_state_dict(torch.load(weight_file_path))
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, label]
    pred_score.backward()
    gradients = img.grad
    channel_R = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_G = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_B = gradients[0, 2, :, :].cpu().detach().numpy()

    return channel_R, channel_G, channel_B


def compute_jacobian(image_path, weight_file_path, model_cnn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = Image.open(image_path)
    image = data_transform(image)
    image = torch.unsqueeze(image, dim=0).to(device)
    var_image = image.clone().detach()
    var_image.requires_grad = True

    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_file_path))
    model.eval()
    output = model(var_image)

    # Initialize jacobian with the same shape as (output.shape[1], image.shape)
    jacobian = torch.zeros((output.shape[1], *var_image.shape[1:]), device=device)

    for i in range(output.shape[1]):  # Compute Jacobian for all classes
        if var_image.grad is not None:
            var_image.grad.zero_()
        output[0][i].backward(retain_graph=True)
        if var_image.grad is not None:
            # Copy the derivative to the target place with original shape
            jacobian[i] = var_image.grad.clone().detach()

    return jacobian


def model_predict(image_path, weights_path, model_cnn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weights_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 1000)
    print("Classification value of top-1：", output[top_indices[0]])
    return top_indices[0], top_indices[1]

def model_predict_attack_index(image_path, weights_path, index, model_cnn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weights_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 1000)
    print("Classification value of attack target class：", output[index])
    return output[index]

# The ratio of tampered pixels
ratio = 0.1

# Maximum number of tampered pixels
tampering_elements_max_num = 3 * 224 * 224 * ratio

# Maximum number of iterations
iterations_max_num = int(8 / 255 * (2.4285 - (-2.0357)) / 0.001)

# The number of pixels tampered with per iteration
number_of_tampered_pixels_per_round = math.ceil(tampering_elements_max_num / iterations_max_num)

# Current convolutional neural network model
model_current = googlenet


# The absolute path to store the original image folder
folder_path = r"..."

# Save the folder path for the generated adversarial samples
save_path_for_adversarial_samples = "..."

# Read class_indict
json_path = r"..."
assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

# Load model weights
weights_path = r"..."
assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)


file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)


for file_name in file_list:
    print(file_name)
    actual_img_path = os.path.join(folder_path, file_name)
    # Open the image and perform preprocessing
    image_absolute_path = actual_img_path
    # Open the image file
    image = Image.open(image_absolute_path)
    # Resize the image to the specified size
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Based on the RGB display mode, extract the data of each channel
    actual_image_R_channel = image_array[:, :, 0]
    actual_image_G_channel = image_array[:, :, 1]
    actual_image_B_channel = image_array[:, :, 2]
    # Convert numpy arrays to torch tensors
    actual_image_R_channel = torch.from_numpy(actual_image_R_channel)
    actual_image_G_channel = torch.from_numpy(actual_image_G_channel)
    actual_image_B_channel = torch.from_numpy(actual_image_B_channel)
    actual_image = torch.stack((actual_image_R_channel, actual_image_G_channel, actual_image_B_channel), dim=0)
    actual_image = actual_image.to(torch.float64)

    num_iterations = 0
    mark_matrix_total = torch.ones((3, 224, 224), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    current_label_top1, current_label_top2 = model_predict(actual_img_path, weights_path, model_current)
    target_label = current_label_top2
    actual_label_Top_1 = current_label_top1
    print("Attack target category index：", target_label)
    img_path = actual_img_path

    while num_iterations < iterations_max_num and torch.sum(mark_matrix_total) >= 3 * 224 * 224 * (1 - ratio) and current_label_top1 == actual_label_Top_1:
        num_iterations = num_iterations + 1
        current_label_top1, current_label_top2 = model_predict(img_path, weights_path, model_current)
        print("Current Top-1:", current_label_top1)

        print("Modify the number of elements：", 3 * 224 * 224 - torch.sum(mark_matrix_total))

        target_tmp_R, target_tmp_G, target_tmp_B = saliency_map(img_path, target_label, weights_path, model_current)
        target_tmp_R = torch.from_numpy(target_tmp_R).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        target_tmp_G = torch.from_numpy(target_tmp_G).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        target_tmp_B = torch.from_numpy(target_tmp_B).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        target_tmp = torch.stack((target_tmp_R, target_tmp_G, target_tmp_B), dim=0)

        jacobian_tmp_total = compute_jacobian(img_path, weights_path, model_current)
        sum_tmp = torch.sum(jacobian_tmp_total, dim=0)
        other_sum_tmp = sum_tmp - target_tmp

        mark_matrix_forward = torch.zeros((3, 224, 224)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        mark_matrix_reverse = torch.zeros((3, 224, 224)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        mask_forward = (other_sum_tmp < 0) & (target_tmp > 0)
        mark_matrix_forward[mask_forward] = 1
        other_sum_tmp_forward = other_sum_tmp * mark_matrix_forward

        mask_reverse = (other_sum_tmp > 0) & (target_tmp < 0)
        mark_matrix_reverse[mask_reverse] = 1
        other_sum_tmp_reverse = other_sum_tmp * mark_matrix_reverse


        saliency_map_total_forward = target_tmp * torch.abs(other_sum_tmp_forward) * mark_matrix_forward * mark_matrix_total
        saliency_map_total_reverse = target_tmp * torch.abs(other_sum_tmp_reverse) * mark_matrix_reverse * mark_matrix_total

        flattened_map_forward = saliency_map_total_forward.view(-1)
        top_values_forward, top_indices_forward = torch.topk(flattened_map_forward, k=3*224*224)

        flattened_map_reverse = saliency_map_total_reverse.view(-1)
        top_values_reverse, top_indices_reverse = torch.topk(flattened_map_reverse, k=3*224*224)

        adv_image = data_transform(Image.open(img_path))
        for k in range(number_of_tampered_pixels_per_round):
            coord_forward = torch.tensor(np.unravel_index(top_indices_forward[k].item(), saliency_map_total_forward.shape))
            mark_matrix_total[coord_forward[0], coord_forward[1], coord_forward[2]] = 0
            # actual_image[coord_forward[0], coord_forward[1], coord_forward[2]] = math.ceil(actual_image[coord_forward[0], coord_forward[1], coord_forward[2]] * 1.2)
            actual_image[coord_forward[0], coord_forward[1], coord_forward[2]] = 255
            actual_image = np.clip(actual_image, 0, 255)

            coord_reverse = torch.tensor(np.unravel_index(top_indices_reverse[k].item(), saliency_map_total_reverse.shape))
            mark_matrix_total[coord_reverse[0], coord_reverse[1], coord_reverse[2]] = 0
            # actual_image[coord_reverse[0], coord_reverse[1], coord_reverse[2]] = int(actual_image[coord_reverse[0], coord_reverse[1], coord_reverse[2]] * 0.8)
            actual_image[coord_reverse[0], coord_reverse[1], coord_reverse[2]] = 0
            actual_image = np.clip(actual_image, 0, 255)

        image_rgb = np.stack([actual_image[0], actual_image[1], actual_image[2]], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("attack_Image.png")
        img_path = "attack_Image.png"
        current_label_top1, current_label_top2 = model_predict(img_path, weights_path, model_current)
        print("Target category classification value：", model_predict_attack_index(img_path, weights_path, target_label, model_current))

        if current_label_top1 != actual_label_Top_1:
            current_image_path = "attack_Image.png"
            new_image_path = os.path.join(save_path_for_adversarial_samples, file_name)
            shutil.copy(current_image_path, new_image_path)
            break
        if num_iterations == iterations_max_num or current_label_top1 == actual_label_Top_1:
            current_image_path = "attack_Image.png"
            new_image_path = os.path.join(save_path_for_adversarial_samples, file_name)
            shutil.copy(current_image_path, new_image_path)

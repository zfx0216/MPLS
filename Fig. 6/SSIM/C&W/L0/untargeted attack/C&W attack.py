from CW import CW
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import math


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


def calculate_gradient(r_matrix, g_matrix, b_matrix, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, index]
    pred_score.backward(retain_graph=True)
    gradients = img.grad
    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()
    return channel_r, channel_g, channel_b


def l0_reinput(actual_image_path, adv_channel_R, adv_channel_G, adv_channel_B,
               mark_matrix_R, mark_matrix_G, mark_matrix_B, label, model_cnn, ratio):

    image = Image.open(actual_image_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_channel_R = actual_image[:, :, 0]
    actual_image_channel_R = actual_image_channel_R.astype(np.float64)
    actual_image_channel_G = actual_image[:, :, 1]
    actual_image_channel_G = actual_image_channel_G.astype(np.float64)
    actual_image_channel_B = actual_image[:, :, 2]
    actual_image_channel_B = actual_image_channel_B.astype(np.float64)

    difference_R = np.abs(actual_image_channel_R - adv_channel_R)
    difference_G = np.abs(actual_image_channel_G - adv_channel_G)
    difference_B = np.abs(actual_image_channel_B - adv_channel_B)

    gradient_matrix_R, gradient_matrix_G, gradient_matrix_B = calculate_gradient(
        adv_channel_R, adv_channel_G, adv_channel_B, weights_path, label, model_cnn)

    degree_of_impact_R = np.abs(difference_R * gradient_matrix_R)
    degree_of_impact_G = np.abs(difference_G * gradient_matrix_G)
    degree_of_impact_B = np.abs(difference_B * gradient_matrix_B)

    combined_matrix = np.stack((degree_of_impact_R, degree_of_impact_G, degree_of_impact_B), axis=-1)
    flat_combined_matrix = combined_matrix.flatten()
    min_indices = np.argsort(flat_combined_matrix)[:math.ceil(3 * 224 * 224 * (1 - ratio))]

    if (mark_matrix_R + mark_matrix_G + mark_matrix_B).sum() > int(3 * 224 * 224 * ratio):
        for index in min_indices:
            i, j, channel = np.unravel_index(index, combined_matrix.shape)
            if channel == 0:
                mark_matrix_R[i, j] = 0
            elif channel == 1:
                mark_matrix_G[i, j] = 0
            elif channel == 2:
                mark_matrix_B[i, j] = 0

    for i in range(224):
        for j in range(224):
            if mark_matrix_R[i][j] == 0:
                adv_channel_R[i][j] = actual_image_channel_R[i][j]
            if mark_matrix_G[i][j] == 0:
                adv_channel_G[i][j] = actual_image_channel_G[i][j]
            if mark_matrix_B[i][j] == 0:
                adv_channel_B[i][j] = actual_image_channel_B[i][j]
    return adv_channel_R, adv_channel_G, adv_channel_B


def predict_image_from_rgb_matrices(r_matrix, g_matrix, b_matrix, index_path, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    predicted_class_index = torch.argmax(classification_probability).item()

    return predicted_class_index, output[index].item()


# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# The absolute path to the folder where the original images are stored
folder_path = r"..."

# Save the folder path for the generated adversarial samples
save_path_for_adversarial_samples = "..."

# Absolute path to import weight file
weights_path = "..."

# The allowed tampering_ratio of tampered pixels
tampering_ratio = 0.1

iteration_step_size = 0.001

iterative_num_max = int(8 / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

# The maximum degree and SSIM of tampering of a single pixel
ssim = 0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_current = resnet18

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
    print(file_name)
    success_flag = 0
    image_path = os.path.join(folder_path, file_name)
    image = Image.open(image_path)
    image = image.resize((224, 224))

    image = Image.open(image_path).convert('RGB')

    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_matrix[0] = actual_image[:, :, 0]
    actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
    actual_image_matrix[1] = actual_image[:, :, 1]
    actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
    actual_image_matrix[2] = actual_image[:, :, 2]
    actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

    actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
    actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
    actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

    image = data_transform(image).unsqueeze(0).cuda()

    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load image
    img_absolute_path = image_path
    assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
    img = Image.open(img_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    actual_img = img.to(device)

    # Read class_indict
    json_absolute_path = "..."
    assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

    with open(json_absolute_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = model_current(num_classes=1000).to(device)

    # Load model weights
    weights_absolute_path = weights_path
    assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
    model.load_state_dict(torch.load(weights_absolute_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 1000)

    # When generating adversarial samples for aimless attacks, it is necessary to set the label in "[]" to the true label of the image
    label = torch.tensor([top_indices[0]]).cuda()

    iterative_num = 0
    mark_matrix_R = np.ones((224, 224), dtype=np.float64)
    mark_matrix_G = np.ones((224, 224), dtype=np.float64)
    mark_matrix_B = np.ones((224, 224), dtype=np.float64)

    while(iterative_num < iterative_num_max):
        print("##########################################################")
        print(f"The current number of iterations is {iterative_num}")
        iterative_num = iterative_num + 1

        atk = CW(model, c=100, kappa=0, steps=iterative_num_max, lr=0.01, ssim=ssim)

        adv_image = atk(image, label, actual_image_transform_matrix, actual_image_matrix)
        image_rgb = adv_image

        iterative_image_top1_label, x = predict_image_from_rgb_matrices(image_rgb[0], image_rgb[1], image_rgb[2], json_absolute_path, weights_absolute_path, 0, model_current)
        if iterative_image_top1_label != label:
            image_rgb[0], image_rgb[1], image_rgb[2] = l0_reinput(
                image_path, image_rgb[0], image_rgb[1], image_rgb[2], mark_matrix_R, mark_matrix_G, mark_matrix_B,
                label, model_current, tampering_ratio)

            current_label, x = predict_image_from_rgb_matrices(
                image_rgb[0], image_rgb[1], image_rgb[2], json_absolute_path, weights_path, 0, model_current
            )
            if current_label != label:
                bset_adv_R = image_rgb[0].copy()
                bset_adv_G = image_rgb[1].copy()
                bset_adv_B = image_rgb[2].copy()
                success_flag = 1
                break
        if iterative_image_top1_label == label and iterative_num == iterative_num_max:

            image_rgb[0], image_rgb[1], image_rgb[2] = l0_reinput(
                image_path, image_rgb[0], image_rgb[1], image_rgb[2], mark_matrix_R, mark_matrix_G, mark_matrix_B,
                label, model_current, tampering_ratio)

            bset_adv_R = image_rgb[0].copy()
            bset_adv_G = image_rgb[1].copy()
            bset_adv_B = image_rgb[2].copy()
            break

        img_array = np.stack([image_rgb[0], image_rgb[1], image_rgb[2]], axis=-1).astype(np.uint8)
        img = Image.fromarray(img_array)
        image = data_transform(img)
        image = torch.unsqueeze(image, dim=0)

    # Combine three channels into an RGB image
    image_rgb = np.stack([bset_adv_R, bset_adv_G, bset_adv_B], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    image_pil.save(new_image_path)
"""
This code file is used for targeted attacks on a single image based on AlexNet with specified probabilities
Step 1: You need to set the 'outputting directory' parameter to the absolute path of the folder where the final results are saved
Step 2: You need to set "act_image_absolutepath" to the absolute path address of the original image
Step 3: You need to set the parameter "index_file_obsolutepath" to the absolute path address of the AlexNet weight file
Finally execute it
"""


import json
import shutil
import matplotlib.pyplot as plt
from torchvision.models import alexnet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

"""
Optional, whether to generate targeted adversarial samples with a specified probability
If set to "True", it generates a targeted adversarial sample pattern with a specified probability. 
If set to "None", it is not set to generate a targeted adversarial sample pattern with a specified probability
"""
classification_probability = True

# The maximum degree of tampering of a single pixel
degree = 8

# Specify the index number of the attack category with a target attack
attack_index = 455

# The absolute path to the folder where adversarial samples are saved
output_directory = "..."
###########################################################################################
# Perform category prediction on the actual image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialization operations
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Load image
actual_image_absolute_path = "..."
assert os.path.exists(actual_image_absolute_path), "file: '{}' dose not exist.".format(actual_image_absolute_path)
img = Image.open(actual_image_absolute_path)
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
actual_image = np.array(image)

# The R, G, and B three channel matrix of the actual image
actual_image_channel_R = actual_image[:, :, 0]
actual_image_channel_G = actual_image[:, :, 1]
actual_image_channel_B = actual_image[:, :, 2]

# The absolute path of the model index file
index_file_absolute_path = "..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)

with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# Create model
model = alexnet(num_classes=1000).to(device)

# Load model weights
weight_file_absolute_path = "..."
assert os.path.exists(weight_file_absolute_path), "file: '{}' dose not exist.".format(weight_file_absolute_path)
model.load_state_dict(torch.load(weight_file_absolute_path))

# Set the model to evaluation mode
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)

top_probs, top_indices = torch.topk(predict, 3)
print("Top-1 category index number of the actual image:")
print(top_indices[0])

# Record the index number of the initial Top-1 category of the image
actual_image_index = top_indices[0]
###################################################################################
flag = 0
iterative_image_path = actual_image_absolute_path

# Iteration tampering frequency
num_iterative = 0

while flag == 0:
    num_iterative = num_iterative + 1
    img = Image.open(iterative_image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Create model
    model = alexnet(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_file_absolute_path))
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(predict, 3)

    print("The Top-1 category index number of the current iteration image：")
    print(top_indices[0])

    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, top_indices[0]]
    pred_score.backward(retain_graph=True)
    gradients = img.grad

    pixel_weight_matrix_R = gradients[0, 0, :, :].cpu().detach().numpy()
    pixel_weight_matrix_G = gradients[0, 1, :, :].cpu().detach().numpy()
    pixel_weight_matrix_B = gradients[0, 2, :, :].cpu().detach().numpy()

    np.savetxt("iterative_sample_pixel_weight_matrix_top1_R.txt", pixel_weight_matrix_R, fmt="%.10f", delimiter=" ")
    np.savetxt("iterative_sample_pixel_weight_matrix_top1_G.txt", pixel_weight_matrix_G, fmt="%.10f", delimiter=" ")
    np.savetxt("iterative_sample_pixel_weight_matrix_top1_B.txt", pixel_weight_matrix_B, fmt="%.10f", delimiter=" ")

   ###########################################################################################################
    # Calculate the pixel weight matrix of the attack category for the current iterative image
    img = Image.open(iterative_image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model = alexnet(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_file_absolute_path))
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output_attack = model(img)
    pred_score_attack = output_attack[0, attack_index]
    pred_score_attack.backward(retain_graph=True)
    gradients_attack = img.grad

    pixel_weight_matrix_attack_R = gradients_attack[0, 0, :, :].cpu().detach().numpy()
    pixel_weight_matrix_attack_G = gradients_attack[0, 1, :, :].cpu().detach().numpy()
    pixel_weight_matrix_attack_B = gradients_attack[0, 2, :, :].cpu().detach().numpy()

    np.savetxt("iterative_sample_pixel_weight_matrix_attack_R.txt", pixel_weight_matrix_attack_R, fmt="%.10f", delimiter=" ")
    np.savetxt("iterative_sample_pixel_weight_matrix_attack_G.txt", pixel_weight_matrix_attack_G, fmt="%.10f", delimiter=" ")
    np.savetxt("iterative_sample_pixel_weight_matrix_attack_B.txt", pixel_weight_matrix_attack_B, fmt="%.10f", delimiter=" ")

    ################################################################################################################
    # Starting to tamper with
    image = Image.open(iterative_image_path)
    image = image.resize((224, 224))
    iterative_image = np.array(image)

    # The RGB three channel matrix of the image in this iteration
    iterative_image_channel_R = iterative_image[:, :, 0]
    iterative_image_channel_G = iterative_image[:, :, 1]
    iterative_image_channel_B = iterative_image[:, :, 2]

    iterative_image_pixel_weight_matrix_for_top_1_ategories_path_R = "iterative_sample_pixel_weight_matrix_top1_R.txt"
    iterative_image_pixel_weight_matrix_for_top_1_ategories_path_G = "iterative_sample_pixel_weight_matrix_top1_G.txt"
    iterative_image_pixel_weight_matrix_for_top_1_ategories_path_B = "iterative_sample_pixel_weight_matrix_top1_B.txt"

    iterative_image_pixel_weight_matrix_for_Top_1_R = read_matrix(iterative_image_pixel_weight_matrix_for_top_1_ategories_path_R)
    iterative_image_pixel_weight_matrix_for_Top_1_G = read_matrix(iterative_image_pixel_weight_matrix_for_top_1_ategories_path_G)
    iterative_image_pixel_weight_matrix_for_Top_1_B = read_matrix(iterative_image_pixel_weight_matrix_for_top_1_ategories_path_B)

    # Calculate the pixel classification contribution matrix of the R, G, and B channels in the Top-1 category for the image that has been tampered with in the current iteration
    pixel_classification_contribution_matrix_Top_1_R = ((iterative_image_channel_R / 255) - 0.485) / 0.229 * iterative_image_pixel_weight_matrix_for_Top_1_R
    pixel_classification_contribution_matrix_Top_1_G = ((iterative_image_channel_G / 255) - 0.456) / 0.224 * iterative_image_pixel_weight_matrix_for_Top_1_G
    pixel_classification_contribution_matrix_Top_1_B = ((iterative_image_channel_B / 255) - 0.406) / 0.225 * iterative_image_pixel_weight_matrix_for_Top_1_B


    pixel_weight_matrix_for_attack_ategories_path_R = "iterative_sample_pixel_weight_matrix_attack_R.txt"
    pixel_weight_matrix_for_attack_ategories_path_G = "iterative_sample_pixel_weight_matrix_attack_G.txt"
    pixel_weight_matrix_for_attack_ategories_path_B = "iterative_sample_pixel_weight_matrix_attack_B.txt"

    iterative_image_pixel_weight_matrix_for_attack_category_R = read_matrix(pixel_weight_matrix_for_attack_ategories_path_R)
    iterative_image_pixel_weight_matrix_for_attack_category_G = read_matrix(pixel_weight_matrix_for_attack_ategories_path_G)
    iterative_image_pixel_weight_matrix_for_attack_category_B = read_matrix(pixel_weight_matrix_for_attack_ategories_path_B)

    # # Calculate the standardized R, G, and B three channel matrix of the tampered image in the current iteration
    iterative_image_standardized_matrix_R = ((iterative_image_channel_R / 255) - 0.485) / 0.229
    iterative_image_standardized_matrix_G = ((iterative_image_channel_G / 255) - 0.456) / 0.224
    iterative_image_standardized_matrix_B = ((iterative_image_channel_B / 255) - 0.406) / 0.225

    # Iteration step size
    ratio = 0.01
    ####################################################################################################
    standardized_matrix_R_increase = iterative_image_standardized_matrix_R.copy()
    standardized_matrix_G_increase = iterative_image_standardized_matrix_G.copy()
    standardized_matrix_B_increase = iterative_image_standardized_matrix_B.copy()

    addend = 0
    for i in range(224):
        for j in range(224):
            addend = abs(iterative_image_standardized_matrix_R[i][j] * ratio)
            standardized_matrix_R_increase[i][j] = standardized_matrix_R_increase[i][j] + addend

            addend = abs(iterative_image_standardized_matrix_G[i][j] * ratio)
            standardized_matrix_G_increase[i][j] = standardized_matrix_G_increase[i][j] + addend

            addend = abs(iterative_image_standardized_matrix_B[i][j] * ratio)
            standardized_matrix_B_increase[i][j] = standardized_matrix_B_increase[i][j] + addend
    ####################################################################################################
    standardized_matrix_R_decrease = iterative_image_standardized_matrix_R.copy()
    standardized_matrix_G_decrease = iterative_image_standardized_matrix_G.copy()
    standardized_matrix_B_decrease = iterative_image_standardized_matrix_B.copy()

    addend = 0
    for i in range(224):
        for j in range(224):
            addend = abs(iterative_image_standardized_matrix_R[i][j] * ratio)
            standardized_matrix_R_decrease[i][j] = standardized_matrix_R_decrease[i][j] - addend

            addend = abs(iterative_image_standardized_matrix_G[i][j] * ratio)
            standardized_matrix_G_decrease[i][j] = standardized_matrix_G_decrease[i][j] - addend

            addend = abs(iterative_image_standardized_matrix_B[i][j] * ratio)
            standardized_matrix_B_decrease[i][j] = standardized_matrix_B_decrease[i][j] - addend

    ########################################################################################################################
    change_matrix_R = iterative_image_channel_R.copy()
    change_matrix_R = change_matrix_R.astype(np.float64)

    change_matrix_G = iterative_image_channel_G.copy()
    change_matrix_G = change_matrix_G.astype(np.float64)

    change_matrix_B = iterative_image_channel_B.copy()
    change_matrix_B = change_matrix_B.astype(np.float64)

    for i in range(224):
        for j in range(224):
            #################################################################################################################################################################################################
            if iterative_image_standardized_matrix_R[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0 and pixel_classification_contribution_matrix_Top_1_R[i][j] < 0:
                change_matrix_R[i][j] = math.ceil((standardized_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] > 0 and pixel_classification_contribution_matrix_Top_1_G[i][j] < 0:
                change_matrix_G[i][j] = math.ceil((standardized_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0 and pixel_classification_contribution_matrix_Top_1_B[i][j] < 0:
                change_matrix_B[i][j] = math.ceil((standardized_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
            #########################################################################################################
            if iterative_image_standardized_matrix_R[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] < 0 and pixel_classification_contribution_matrix_Top_1_R[i][j] < 0:
                change_matrix_R[i][j] = int((standardized_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] < 0 and pixel_classification_contribution_matrix_Top_1_G[i][j] < 0:
                change_matrix_G[i][j] = int((standardized_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] < 0 and pixel_classification_contribution_matrix_Top_1_B[i][j] < 0:
                change_matrix_B[i][j] = int((standardized_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
            ########################################################################################################
            if iterative_image_standardized_matrix_R[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0 and pixel_classification_contribution_matrix_Top_1_R[i][j] > 0:
                change_matrix_R[i][j] = math.ceil((standardized_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] > 0 and pixel_classification_contribution_matrix_Top_1_G[i][j] > 0:
                change_matrix_G[i][j] = math.ceil((standardized_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] > 0 and pixel_classification_contribution_matrix_Top_1_B[i][j] > 0:
                change_matrix_B[i][j] = math.ceil((standardized_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
            ###########################################################################################################
            if iterative_image_standardized_matrix_R[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] < 0 and pixel_classification_contribution_matrix_Top_1_R[i][j] > 0:
                change_matrix_R[i][j] = int((standardized_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                actual_image_channel_R[i][j] + degree)
                change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

            if iterative_image_standardized_matrix_G[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] < 0 and pixel_classification_contribution_matrix_Top_1_G[i][j] > 0:
                change_matrix_G[i][j] = int((standardized_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                actual_image_channel_G[i][j] + degree)
                change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

            if iterative_image_standardized_matrix_B[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] < 0 and pixel_classification_contribution_matrix_Top_1_B[i][j] > 0:
                change_matrix_B[i][j] = int((standardized_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                actual_image_channel_B[i][j] + degree)
                change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)


    # Generate images that have been tampered with in this iteration
    image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
    image_rgb = image_rgb.astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)
    image_pil.save("Targeted attack images.png")

    # This round of image tampering has been completed, input the model to determine whether the tampering was successful
    model = alexnet(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_file_absolute_path))

    iterative_image_path = "Targeted attack images.png"
    img = Image.open(iterative_image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 3)

    iterative_image_index = top_indices[0]

    if iterative_image_index == attack_index:
        print("Successful target attack:")
        for i in range(3):
            class_index = top_indices[i].item()
            class_prob = top_probs[i].item()
            print(
                "Top {}: index: {}  class: {:10}   Classification probability: {:.5f}   Classification value: {:.6f}".format(
                    i + 1, class_index, class_indict[str(class_index)],
                    class_prob, output[class_index].numpy()))

        current_image_path = "Targeted attack images.png"
        new_image_name = "successful_target_attack.png"
        new_image_path = os.path.join(output_directory, new_image_name)
        shutil.copy(current_image_path, new_image_path)
        flag = 1
        print("Iteration tampering frequency：")
        print(num_iterative)
    elif iterative_image_index != attack_index:
        print("This round of target attack failed！")
        iterative_image_path = "Targeted attack images.png"
        flag = 0

if classification_probability == True:
    actual_image_absolute_path = new_image_path
    assert os.path.exists(actual_image_absolute_path), "file: '{}' dose not exist.".format(actual_image_absolute_path)
    img = Image.open(actual_image_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)

    with open(index_file_absolute_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = alexnet(num_classes=1000).to(device)

    # Load model weights
    assert os.path.exists(weight_file_absolute_path), "file: '{}' dose not exist.".format(weight_file_absolute_path)
    model.load_state_dict(torch.load(weight_file_absolute_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)

    top_probs, top_indices = torch.topk(predict, 3)
    print("The Top-1 category index number of the current iteration image：")
    print(top_indices[0])

    actual_image_index = top_indices[0]

    image = Image.open(actual_image_absolute_path)
    input_tensor = data_transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    attack_index = top_indices[0]
    ############################################################################
    flag = 0
    iterative_image_path = actual_image_absolute_path
    designated_probability_value = 0.5
    while flag == 0 or designated_probability_value < 0.9:
        num_iterative = num_iterative + 1
        img = Image.open(iterative_image_path)
        plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        model = alexnet(num_classes=1000).to(device)
        model.load_state_dict(torch.load(weight_file_absolute_path))

        img = img.to(device)
        model.eval()
        img.requires_grad_()
        output_attack = model(img)
        pred_score_attack = output_attack[0, attack_index]
        pred_score_attack.backward(retain_graph=True)
        gradients_attack = img.grad

        pixel_weight_matrix_strengthen_R = gradients_attack[0, 0, :, :].cpu().detach().numpy()
        pixel_weight_matrix_strengthen_G = gradients_attack[0, 1, :, :].cpu().detach().numpy()
        pixel_weight_matrix_strengthen_B = gradients_attack[0, 2, :, :].cpu().detach().numpy()

        np.savetxt("iterative_sample_pixel_weight_matrix_strengthen_R.txt", pixel_weight_matrix_strengthen_R, fmt="%.10f", delimiter=" ")
        np.savetxt("iterative_sample_pixel_weight_matrix_strengthen_G.txt", pixel_weight_matrix_strengthen_G, fmt="%.10f", delimiter=" ")
        np.savetxt("iterative_sample_pixel_weight_matrix_strengthen_B.txt", pixel_weight_matrix_strengthen_B, fmt="%.10f", delimiter=" ")

        ###########################################################################################################
        image = Image.open(iterative_image_path)
        image = image.resize((224, 224))
        iterative_image = np.array(image)

        iterative_image_channel_R = iterative_image[:, :, 0]
        iterative_image_channel_G = iterative_image[:, :, 1]
        iterative_image_channel_B = iterative_image[:, :, 2]

        pixel_weight_matrix_for_attack_ategories_path_R = "iterative_sample_pixel_weight_matrix_strengthen_R.txt"
        pixel_weight_matrix_for_attack_ategories_path_G = "iterative_sample_pixel_weight_matrix_strengthen_G.txt"
        pixel_weight_matrix_for_attack_ategories_path_B = "iterative_sample_pixel_weight_matrix_strengthen_B.txt"

        iterative_image_pixel_weight_matrix_for_attack_category_R = read_matrix(pixel_weight_matrix_for_attack_ategories_path_R)
        iterative_image_pixel_weight_matrix_for_attack_category_G = read_matrix(pixel_weight_matrix_for_attack_ategories_path_G)
        iterative_image_pixel_weight_matrix_for_attack_category_B = read_matrix(pixel_weight_matrix_for_attack_ategories_path_B)

        iterative_image_standardized_matrix_R = ((iterative_image_channel_R / 255) - 0.485) / 0.229
        iterative_image_standardized_matrix_G = ((iterative_image_channel_G / 255) - 0.456) / 0.224
        iterative_image_standardized_matrix_B = ((iterative_image_channel_B / 255) - 0.406) / 0.225

        num = 0
        # Iteration step size
        ratio = 0.01
        #############################################################################################
        standardized_matrix_R_increase = iterative_image_standardized_matrix_R.copy()
        standardized_matrix_G_increase = iterative_image_standardized_matrix_G.copy()
        standardized_matrix_B_increase = iterative_image_standardized_matrix_B.copy()

        addend = 0
        for i in range(224):
            for j in range(224):
                addend = abs(iterative_image_standardized_matrix_R[i][j] * ratio)
                standardized_matrix_R_increase[i][j] = standardized_matrix_R_increase[i][j] + addend

                addend = abs(iterative_image_standardized_matrix_G[i][j] * ratio)
                standardized_matrix_G_increase[i][j] = standardized_matrix_G_increase[i][j] + addend

                addend = abs(iterative_image_standardized_matrix_B[i][j] * ratio)
                standardized_matrix_B_increase[i][j] = standardized_matrix_B_increase[i][j] + addend
        ##############################################################################################
        standardized_matrix_R_decrease = iterative_image_standardized_matrix_R.copy()
        standardized_matrix_G_decrease = iterative_image_standardized_matrix_G.copy()
        standardized_matrix_B_decrease = iterative_image_standardized_matrix_B.copy()

        addend = 0
        for i in range(224):
            for j in range(224):
                addend = abs(iterative_image_standardized_matrix_R[i][j] * ratio)
                standardized_matrix_R_decrease[i][j] = standardized_matrix_R_decrease[i][j] - addend

                addend = abs(iterative_image_standardized_matrix_G[i][j] * ratio)
                standardized_matrix_G_decrease[i][j] = standardized_matrix_G_decrease[i][j] - addend

                addend = abs(iterative_image_standardized_matrix_B[i][j] * ratio)
                standardized_matrix_B_decrease[i][j] = standardized_matrix_B_decrease[i][j] - addend

        ########################################################################################################################
        change_matrix_R = iterative_image_channel_R.copy()
        change_matrix_R = change_matrix_R.astype(np.float64)

        change_matrix_G = iterative_image_channel_G.copy()
        change_matrix_G = change_matrix_G.astype(np.float64)

        change_matrix_B = iterative_image_channel_B.copy()
        change_matrix_B = change_matrix_B.astype(np.float64)

        for i in range(224):
            for j in range(224):
                ##################################################################################################################################
                if iterative_image_standardized_matrix_R[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0:
                    change_matrix_R[i][j] = math.ceil((standardized_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)


                if iterative_image_standardized_matrix_G[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] > 0:
                    change_matrix_G[i][j] = math.ceil((standardized_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

                if iterative_image_standardized_matrix_B[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] > 0:
                    change_matrix_B[i][j] = math.ceil((standardized_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)

                #########################################################################################################

                if iterative_image_standardized_matrix_R[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] < 0:
                    change_matrix_R[i][j] = int((standardized_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

                if iterative_image_standardized_matrix_G[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] < 0:
                    change_matrix_G[i][j] = int((standardized_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

                if iterative_image_standardized_matrix_B[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] < 0:
                    change_matrix_B[i][j] = int((standardized_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                ########################################################################################################
                if iterative_image_standardized_matrix_R[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] > 0:
                    change_matrix_R[i][j] = math.ceil((standardized_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

                if iterative_image_standardized_matrix_G[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] > 0:
                    change_matrix_G[i][j] = math.ceil((standardized_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

                if iterative_image_standardized_matrix_B[i][j] < 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] > 0:
                    change_matrix_B[i][j] = math.ceil((standardized_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                ###########################################################################################################
                if iterative_image_standardized_matrix_R[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_R[i][j] < 0:
                    change_matrix_R[i][j] = int((standardized_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                    actual_image_channel_R[i][j] + degree)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)

                if iterative_image_standardized_matrix_G[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_G[i][j] < 0:
                    change_matrix_G[i][j] = int((standardized_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                    actual_image_channel_G[i][j] + degree)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)

                if iterative_image_standardized_matrix_B[i][j] > 0 and iterative_image_pixel_weight_matrix_for_attack_category_B[i][j] < 0:
                    change_matrix_B[i][j] = int((standardized_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                    actual_image_channel_B[i][j] + degree)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)

        image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("Targeted attack images.png")

        # Create model
        model = alexnet(num_classes=1000).to(device)
        model.load_state_dict(torch.load(weight_file_absolute_path))
        iterative_image_path = "Targeted attack images.png"
        img = Image.open(iterative_image_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            top_probs, top_indices = torch.topk(predict, 3)

        iterative_image_index = top_indices[0]
        designated_probability_value = top_probs[0]

        if iterative_image_index == attack_index:
            for i in range(3):
                class_index = top_indices[i].item()
                class_prob = top_probs[i].item()
                print(
                    "Top {}: index: {}  class: {:10}   Classification probability: {:.5f}   Classification value: {:.6f}".format(
                        i + 1, class_index, class_indict[str(class_index)],
                        class_prob, output[class_index].numpy()))

            current_image_path = "Targeted attack images.png"
            new_image_name = "Adversarial sample.png"
            new_image_path = os.path.join(output_directory, new_image_name)
            shutil.copy(current_image_path, new_image_path)
            flag = 1
            print("Iteration tampering frequency：")
            print(num_iterative)
        elif iterative_image_index != attack_index:
            print("This round of target attack failed！")
            iterative_image_path = "Targeted attack images.png"
            flag = 0
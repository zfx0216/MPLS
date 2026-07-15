import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet
import numpy as np
import math
import shutil
import torch.nn as nn
import z3
from skimage.metrics import structural_similarity as ssim
from skimage import io

def calculate_gradient(image_path, label, weight_path, model_cnn):
    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = data_transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Load model
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Define loss
    loss = nn.CrossEntropyLoss()
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)

    # Forward pass
    outputs = model(img_tensor)

    # Calculate loss
    cost = loss(outputs, label_tensor)

    # Calculate gradient
    grad = torch.autograd.grad(cost, img_tensor, retain_graph=False, create_graph=False)[0]

    return grad.squeeze(0).cpu().numpy()

def divide_matrix(input_matrix, level):
    unique_elements = np.unique(input_matrix)
    num_unique = len(unique_elements)
    threshold = np.percentile(unique_elements, level)
    mark_matrix_actual_index = np.where(input_matrix > threshold, 1, 0)
    return mark_matrix_actual_index


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()
    return(predicted_class_index,output[index])

def create_mask_matrix(gradient_matrix, number):
    if gradient_matrix.shape != (3, 224, 224):
        raise ValueError("The shape of the input matrix must be (3, 224, 224)")
    abs_matrix = np.abs(gradient_matrix)
    flattened = abs_matrix.flatten()
    sorted_indices = np.argsort(flattened)[::-1]
    top_indices = sorted_indices[:number]
    mask_l0 = np.zeros_like(gradient_matrix, dtype=np.uint8)
    mask_l0.flat[top_indices] = 1
    return mask_l0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# The absolute path of index file
index_file_absolute_path = r"..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = r"..."

# The maximum degree of tampering of a single pixel
degree = 255

# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

tampering_quantity = int(0.02 * 3 * 224 * 224)

model_current = alexnet

ssim_threshold = 0.99

# Set the number of perturbed pixels
perturbed_pixel_ratio = 1

# Import the folder path where the original input images are stored.
actual_images_folder_absolute_path = r"..."

# Import the path of the folder where the generated adversarial examples are saved.
output_directory = r"..."

# The total number of images in the folder that require tampering attacks
the_total_number_of_tampered_images = 0

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

# Record the total number of original images
image_num = 0

# Record the total number of successfully attacked images
success_num = 0

for image_file in image_files:
    print(f"The image number currently being processed is:{image_file}")
    image_num = image_num + 1
    ratio = 1 - perturbed_pixel_ratio

    the_total_number_of_tampered_images = the_total_number_of_tampered_images + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_channel_R = actual_image[:, :, 0]
    actual_image_channel_R = actual_image_channel_R.astype(np.float64)
    actual_image_channel_G = actual_image[:, :, 1]
    actual_image_channel_G = actual_image_channel_G.astype(np.float64)
    actual_image_channel_B = actual_image[:, :, 2]
    actual_image_channel_B = actual_image_channel_B.astype(np.float64)

    actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float32)
    actual_image_matrix[0] = actual_image_channel_R
    actual_image_matrix[1] = actual_image_channel_G
    actual_image_matrix[2] = actual_image_channel_B

    # Standardize the actual image
    actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
    actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
    actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225
    ####################################################################################################
    # Perform category prediction on the original image
    actual_image_index, x = predict_image_path(actual_image_absolute_path, index_file_absolute_path,
                                               weight_file_absolute_path, 0, model_current)
    print(f"Top-1 category index number of the actual image:{actual_image_index}")

    gradient_matrix = calculate_gradient(actual_image_absolute_path, actual_image_index, weight_file_absolute_path, model_current)
    mask_l0 = create_mask_matrix(gradient_matrix, tampering_quantity)

    flag = 0
    num_iterative = 0
    iterative_image_path = actual_image_absolute_path
    while flag == 0 and num_iterative <= max_num_iterative:
        num_iterative = num_iterative + 1
        print("##########################################################################")
        print(f"Current iteration count:{num_iterative}")
        ###################################################################
        image = Image.open(iterative_image_path)
        image = image.resize((224, 224))
        iterative_image = np.array(image)

        # The R, G, B three channel matrix of the actual image
        iterative_image_channel_R = iterative_image[:, :, 0]
        iterative_image_channel_R = iterative_image_channel_R.astype(np.float64)
        iterative_image_channel_G = iterative_image[:, :, 1]
        iterative_image_channel_G = iterative_image_channel_G.astype(np.float64)
        iterative_image_channel_B = iterative_image[:, :, 2]
        iterative_image_channel_B = iterative_image_channel_B.astype(np.float64)

        # Standardize the actual image
        iterative_image_transform_matrix_R = ((iterative_image_channel_R / 255) - 0.485) / 0.229
        iterative_image_transform_matrix_G = ((iterative_image_channel_G / 255) - 0.456) / 0.224
        iterative_image_transform_matrix_B = ((iterative_image_channel_B / 255) - 0.406) / 0.225
        ####################################################################################################
        # Calculate the Top-1 category of the iterative image in this round
        iterative_image_top1_index, x = predict_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, actual_image_index, model_current)
        print(f"Top-1 category index number of the actual image:{iterative_image_top1_index}")
        ####################################################################################
        gradient_matrix = calculate_gradient(iterative_image_path, actual_image_index, weight_file_absolute_path, model_current)
        iterative_image_pixel_weight_matrix_in_actual_label_R = gradient_matrix[0]
        iterative_image_pixel_weight_matrix_in_actual_label_G = gradient_matrix[1]
        iterative_image_pixel_weight_matrix_in_actual_label_B = gradient_matrix[2]

        mark_matrix_final = divide_matrix(np.abs(gradient_matrix), ratio)
        mark_matrix_final_R = mark_matrix_final[0]
        mark_matrix_final_G = mark_matrix_final[1]
        mark_matrix_final_B = mark_matrix_final[2]

        change_matrix_R = iterative_image_channel_R.copy()
        change_matrix_G = iterative_image_channel_G.copy()
        change_matrix_B = iterative_image_channel_B.copy()
        num = 0

        solver = z3.Solver()
        solver.set(logic='QF_FP')

        step_val_fp = z3.FPVal(float(iteration_step_size), z3.Float64())
        neg_step_val_fp = z3.FPVal(-float(iteration_step_size), z3.Float64())

        mask_r = mask_l0[0]
        mask_g = mask_l0[1]
        mask_b = mask_l0[2]

        z3_vars_r = [[None for _ in range(224)] for _ in range(224)]
        z3_vars_g = [[None for _ in range(224)] for _ in range(224)]
        z3_vars_b = [[None for _ in range(224)] for _ in range(224)]

        total_marked_pixels = 0

        rows_r, cols_r = np.where(mask_r == 1)
        for i, j in zip(rows_r, cols_r):
            g = iterative_image_pixel_weight_matrix_in_actual_label_R[i][j]
            if g == 0:
                continue
            total_marked_pixels += 1
            var = z3.FP(f'd_r_{i}_{j}', z3.Float64())
            z3_vars_r[i][j] = var
            if g > 0:
                solver.add(var == step_val_fp)
            else:
                solver.add(var == neg_step_val_fp)

        rows_g, cols_g = np.where(mask_g == 1)
        for i, j in zip(rows_g, cols_g):
            g = iterative_image_pixel_weight_matrix_in_actual_label_G[i][j]
            if g == 0:
                continue
            total_marked_pixels += 1
            var = z3.FP(f'd_g_{i}_{j}', z3.Float64())
            z3_vars_g[i][j] = var
            if g > 0:
                solver.add(var == step_val_fp)
            else:
                solver.add(var == neg_step_val_fp)


        rows_b, cols_b = np.where(mask_b == 1)
        for i, j in zip(rows_b, cols_b):
            g = iterative_image_pixel_weight_matrix_in_actual_label_B[i][j]
            if g == 0:
                continue
            total_marked_pixels += 1
            var = z3.FP(f'd_b_{i}_{j}', z3.Float64())
            z3_vars_b[i][j] = var
            if g > 0:
                solver.add(var == step_val_fp)
            else:
                solver.add(var == neg_step_val_fp)

        if solver.check() == z3.sat:
            z3_model = solver.model()

            for i, j in zip(rows_r, cols_r):
                var = z3_vars_r[i][j]
                if var is not None:
                    val_num = z3_model.eval(z3.fpToReal(z3_model[var]))
                    delta_float = float(val_num.as_fraction().numerator) / float(val_num.as_fraction().denominator)

                    iterative_image_transform_matrix_R[i][j] += delta_float
                    pixel_expr = (iterative_image_transform_matrix_R[i][j] * 0.229 + 0.485) * 255
                    g = iterative_image_pixel_weight_matrix_in_actual_label_R[i][j]
                    change_matrix_R[i][j] = math.ceil(pixel_expr) if g > 0 else int(pixel_expr)

            for i, j in zip(rows_g, cols_g):
                var = z3_vars_g[i][j]
                if var is not None:
                    val_num = z3_model.eval(z3.fpToReal(z3_model[var]))
                    delta_float = float(val_num.as_fraction().numerator) / float(val_num.as_fraction().denominator)

                    iterative_image_transform_matrix_G[i][j] += delta_float
                    pixel_expr = (iterative_image_transform_matrix_G[i][j] * 0.224 + 0.456) * 255
                    g = iterative_image_pixel_weight_matrix_in_actual_label_G[i][j]
                    change_matrix_G[i][j] = math.ceil(pixel_expr) if g > 0 else int(pixel_expr)

            for i, j in zip(rows_b, cols_b):
                var = z3_vars_b[i][j]
                if var is not None:
                    val_num = z3_model.eval(z3.fpToReal(z3_model[var]))
                    delta_float = float(val_num.as_fraction().numerator) / float(val_num.as_fraction().denominator)

                    iterative_image_transform_matrix_B[i][j] += delta_float
                    pixel_expr = (iterative_image_transform_matrix_B[i][j] * 0.225 + 0.406) * 255
                    g = iterative_image_pixel_weight_matrix_in_actual_label_B[i][j]
                    change_matrix_B[i][j] = math.ceil(pixel_expr) if g > 0 else int(pixel_expr)
        else:
            print("Warning")

        change_matrix_R = np.clip(change_matrix_R, 0, 255)
        change_matrix_G = np.clip(change_matrix_G, 0, 255)
        change_matrix_B = np.clip(change_matrix_B, 0, 255)
        change_matrix = np.zeros((3, 224, 224), dtype=np.float32)
        change_matrix[0] = change_matrix_R
        change_matrix[1] = change_matrix_G
        change_matrix[2] = change_matrix_B

        change_matrix[mask_l0 == 0] = actual_image_matrix[mask_l0 == 0]

        # Generate images that have been tampered with in this iteration
        image_rgb = np.stack([change_matrix[0], change_matrix[1], change_matrix[2]],axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("target attack image.png")

        img_ori = actual_image_matrix.transpose(1, 2, 0).astype(np.uint8)
        img_adv = change_matrix.transpose(1, 2, 0).astype(np.uint8)

        current_ssim = ssim(img_ori, img_adv, win_size=3, data_range=255)
        print(f"Current SSIM: {current_ssim:.4f}")

        if current_ssim < ssim_threshold:
            print(f"SSIM {current_ssim:.4f} < threshold {ssim_threshold}, skip current perturbation")
            iterative_image_path = actual_image_absolute_path
            continue

        # Calculate and query whether the current attack image has been successfully attacked
        attack_image_path = "target attack image.png"
        attack_image_index, x = predict_image_path(attack_image_path, index_file_absolute_path,
                                                   weight_file_absolute_path, actual_image_index, model_current)
        print(f"The top-1 label of the attack image generated in this round：{attack_image_index}")

        if attack_image_index != actual_image_index:
            print("Untargeted attack succeeded.")
            flag = 1
            success_num = success_num + 1
            iterative_image_path = f"{output_directory}\\{image_num}.png"
            shutil.copy(attack_image_path, iterative_image_path)
        else:
            iterative_image_path = f"{output_directory}\\{image_num}.png"
            shutil.copy(attack_image_path, iterative_image_path)
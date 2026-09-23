import torch
from torchvision import transforms
from torchvision.models import vgg11
import os
from PIL import Image
import numpy as np
from fsa_mifgsm import DctMIFGSM
import json
import matplotlib.pyplot as plt

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_current = vgg11

# Create and load model
model = model_current(num_classes=1000).to(device)

# Absolute path to import AlexNet weight file
weights_path = r"..."

assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# data_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
# ])

# The maximum degree and SSIM of tampering of a single pixel
degree = 8

ssim = 0.9
# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

# The absolute path to store the original image folder
folder_path = r"..."

# Save the folder path for the generated adversarial samples
save_path_for_adversarial_samples = r"..."

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
    print(file_name)
    image_path = os.path.join(folder_path, file_name)

    image = Image.open(image_path).convert('RGB')

    # actual image
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

    actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
    actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
    actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
    actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

    image = data_transform(image).unsqueeze(0).cuda()

    # Load image
    img_absolute_path = image_path
    assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
    img = Image.open(img_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Read class_indict
    json_absolute_path = r"..."
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

    # When generating adversarial samples, it is a no target attack, so the true label of the image needs to be set in "[]" here
    label = torch.tensor([top_indices[0]]).cuda()

    atk = DctMIFGSM(model, decay=1, steps=max_num_iterative, alpha=0.001, ssim=ssim)

    adv_image = atk(image, label, actual_image_transform_matrix, actual_image_matrix)

    # Combine three channels into an RGB image
    image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    image_pil.save(new_image_path)

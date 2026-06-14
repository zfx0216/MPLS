import torch
import torchvision.transforms as transforms
from torchvision.models import vgg11
from PIL import Image
import numpy as np
import os
from Autoattack import AutoAttack
import torchattacks
import numpy as np


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):
    adv_image = actual_image_matrix.copy()
    adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.cpu().numpy()
    factors = np.array([[0.229, 0.485], [0.224, 0.456], [0.225, 0.406]])
    scales = np.array([255, 255, 255])
    for c in range(3):
        actual_transform = actual_image_transform_matrix[c]
        adversarial_transform = adversarial_sample_transform_matrix[c]
        actual_image = actual_image_matrix[c]
        mask_greater = adversarial_transform > actual_transform
        mask_less = adversarial_transform < actual_transform
        mask_equal = adversarial_transform == actual_transform
        adv_image[c] = np.where(mask_greater,
                                np.ceil((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                np.where(mask_less,
                                         np.floor((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                         np.where(mask_equal, actual_image, adv_image[c])))

    adv_image = np.clip(adv_image, 0, 255)

    return adv_image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The absolute path to the folder where the original images are stored
folder_path = r"..."

# Save the folder path for the generated attack samples
save_path_for_adversarial_samples = r"..."

weights_path = r'...'

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)


for file_name in file_list:
    print("current",file_name)
    image_path = os.path.join(folder_path, file_name)
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    model = vgg11(pretrained=False)
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

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

    with torch.no_grad():
        pred_label = model(img_tensor).argmax(dim=1).item()


    attack = AutoAttack(
    model,
    norm='L2',
    eps=4 * (8 / 255)
    )

    y_test = torch.tensor([pred_label]).to(device)
    adv_img = attack(img_tensor, y_test)

    with torch.no_grad():
        adv_output = model(adv_img)
        adv_pred = adv_output.argmax(dim=1).item()



    adv_img = adv_img.detach()
    adv_img[0][0] = torch.clamp(adv_img[0][0], min=-2.1179, max=2.2489).detach()
    adv_img[0][1] = torch.clamp(adv_img[0][1], min=-2.0357, max=2.4285).detach()
    adv_img[0][2] = torch.clamp(adv_img[0][2], min=-1.8044, max=2.64).detach()
    x_adv = adv_img.squeeze(0).cpu()

    adv_image = con_transform(actual_image_transform_matrix, x_adv, actual_image_matrix)

    image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    image_pil.save('adv.png')
    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    image_pil.save(new_image_path)
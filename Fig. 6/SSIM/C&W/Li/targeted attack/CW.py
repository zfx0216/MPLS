import torch
import torch.nn as nn
import torch.optim as optim
from attack import Attack
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
from torchvision.models import googlenet
from skimage.metrics import structural_similarity as ssim
from skimage import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def predict_image_from_rgb_matrices(r_matrix, g_matrix, b_matrix, index_path, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    predicted_class_index = torch.argmax(classification_probability).item()

    return predicted_class_index, output[index].item()


def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):
    adv_image = actual_image_matrix.copy()
    adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.cpu().detach().numpy()
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


class CW(Attack):
    def __init__(self, model, c=13, kappa=0, steps=10, lr=0.01, ssim=0.9):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]
        self.ssim = ssim

    def forward(self, images, labels, actual_image_transform_matrix, actual_image_matrix):

        weights_path = "..."
        json_absolute_path = "..."
        model_current = googlenet

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        flag = 0

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        w = images.clone().detach().to(self.device)
        w[0][0] = self.inverse_tanh_space_R(images[0][0]).detach()
        w[0][1] = self.inverse_tanh_space_G(images[0][1]).detach()
        w[0][2] = self.inverse_tanh_space_B(images[0][2]).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_Linf = 999999999
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([w], lr=self.lr)

        adv_image_copy = actual_image_matrix

        for step in range(self.steps):
            print("###########################################################################")

            print(step)
            # Get adversarial images
            adv_images = self.tanh_space(w)
            # Calculate loss
            current_Linf = torch.max(torch.abs(adv_images - images).view(len(images), -1), dim=1)[0]
            Linf_loss = current_Linf.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = Linf_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            adv_images_current = self.tanh_space(w).clone()
            delta = adv_images_current - images

            # Tampering intensity is 4
            delta[0][0] = torch.clamp(adv_images_current[0][0] - images[0][0], min=-0.06849, max=0.06849)
            delta[0][1] = torch.clamp(adv_images_current[0][1] - images[0][1], min=-0.07002, max=0.07002)
            delta[0][2] = torch.clamp(adv_images_current[0][2] - images[0][2], min=-0.06970, max=0.06970)

            adv_images_current[0][0] = torch.clamp(images[0][0] + delta[0][0], min=-2.1179, max=2.2489).detach()
            adv_images_current[0][1] = torch.clamp(images[0][1] + delta[0][1], min=-2.0357, max=2.4285).detach()
            adv_images_current[0][2] = torch.clamp(images[0][2] + delta[0][2], min=-1.804, max=2.64).detach()


            adv_images_current = adv_images_current.squeeze(0).cpu()
            adv_image = adv_images_current.clone()

            adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(adv_image[0],
                                                                            adv_image[1],
                                                                            adv_image[2],
                                                                            json_absolute_path, weights_path, 0,
                                                                            model_current)
            image_np = actual_image_matrix.transpose((1, 2, 0))
            adv_image_np = adv_image.transpose((1, 2, 0))

            SSIM = ssim(image_np, adv_image_np, win_size=3, data_range=255)

            if SSIM < self.ssim:
                return adv_image_copy

            adv_image_copy = adv_image

            if iterative_image_top1_label == labels and best_Linf > current_Linf:
                flag = 1
                best_adv_images = adv_image
                best_Linf = current_Linf
            if flag == 0 and step == 139:
                best_adv_images = adv_image

        return best_adv_images


    def tanh_space(self, x):
        x_clone = x.clone()
        x_clone[0][0] = (0.5 * (torch.tanh(x[0][0]) + 1) - 0.485) / 0.229
        x_clone[0][1] = (0.5 * (torch.tanh(x[0][1]) + 1) - 0.456) / 0.224
        x_clone[0][2] = (0.5 * (torch.tanh(x[0][2]) + 1) - 0.406) / 0.225
        return x_clone

    def inverse_tanh_space_R(self, x):
        converted_value = (x * 0.229 + 0.485) * 2 - 1
        return self.atanh(torch.clamp(converted_value, min=-1, max=1))

    def inverse_tanh_space_G(self, x):
        converted_value = (x * 0.224 + 0.456) * 2 - 1
        return self.atanh(torch.clamp(converted_value, min=-1, max=1))

    def inverse_tanh_space_B(self, x):
        converted_value = (x * 0.225 + 0.406) * 2 - 1
        return self.atanh(torch.clamp(converted_value, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)

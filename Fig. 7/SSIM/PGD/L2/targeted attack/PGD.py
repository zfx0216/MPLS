import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
from attack import Attack
import torchvision.transforms as transforms
import math
import os
from torchvision.models import alexnet
from skimage.metrics import structural_similarity as ssim
from skimage import io

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PGDL2(Attack):
    def __init__(
        self,
        model,
        eps=float(int(np.sqrt(((4 / 255 * (2.4285 + 2.0357)) **2) * 3 * 224 * 224))),
        alpha=0.001,
        steps=10,
        random_start=True,
        eps_for_division=1e-10,
        ssim=0.9
    ):
        super().__init__("PGDL2", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]
        self.ssim = ssim

    def forward(self, images, labels, actual_image_transform_matrix, actual_image_matrix):

        # Absolute path of weight file
        weights_path = r"..."

        # Absolute path of index file
        json_absolute_path = r"..."

        model_current = alexnet
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        print(labels)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images[0][0] = torch.clamp(adv_images[0][0] + delta[0][0], min=-2.1179, max=2.2489).detach()
            adv_images[0][1] = torch.clamp(adv_images[0][1] + delta[0][1], min=-2.0357, max=2.4285).detach()
            adv_images[0][2] = torch.clamp(adv_images[0][2] + delta[0][2], min=-1.8044, max=2.64).detach()

        adv_image_copy = actual_image_matrix

        for _ in range(self.steps):
            _ = _ + 1
            print(_)
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad_norms = (
                torch.norm(grad.view(batch_size, -1), p=2, dim=1)
                + self.eps_for_division
            )  # nopep8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images[0][0] = torch.clamp(images[0][0] + delta[0][0], min=-2.1179, max=2.2489).detach()
            adv_images[0][1] = torch.clamp(images[0][1] + delta[0][1], min=-2.0357, max=2.4285).detach()
            adv_images[0][2] = torch.clamp(images[0][2] + delta[0][2], min=-1.8044, max=2.64).detach()

            adv_image = adv_images.squeeze(0).cpu()
            adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(adv_image[0], adv_image[1], adv_image[2],
                                                                            json_absolute_path, weights_path, 0,
                                                                            model_current)

            image_np = actual_image_matrix.transpose((1, 2, 0))
            adv_image_np = adv_image.transpose((1, 2, 0))

            SSIM = ssim(image_np, adv_image_np, win_size=3, data_range=255)

            if SSIM < self.ssim:
                return adv_image_copy

            adv_image_copy = adv_image

            if iterative_image_top1_label == labels:
                return adv_image

        return adv_image

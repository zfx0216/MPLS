import torch
import torch.nn as nn
import json
from attack import Attack
from dct import dct_2d, idct_2d
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision.models import vgg11
from skimage.metrics import structural_similarity as ssim
from skimage import io

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


class MIFGSM(Attack):

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class DctMIFGSM(Attack):

    def __init__(self, model, eps=8 / 255, alpha=0.001, steps=10, decay=1.0, inside_sign=False, ssim=0.9):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']
        self.inside_sign = inside_sign
        self.ssim = ssim

    def forward(self, images, labels, actual_image_transform_matrix, actual_image_matrix):
        r"""
        Overridden.
        """
        weights_path = r"..."
        json_absolute_path = r"..."

        model_current = vgg11
        # self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        adv_image_copy = actual_image_matrix

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            adv_images_dct = dct_2d(adv_images)
            outputs_dct = self.get_logits(idct_2d(adv_images_dct))


            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
                cost_dct = -loss(outputs_dct, target_labels)
            else:
                cost = loss(outputs, labels)
                cost_dct = loss(outputs_dct, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad_dct = torch.autograd.grad(cost_dct, adv_images_dct,
                                           retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay

            if self.inside_sign:
                grad_dct = grad_dct.sign()
            grad_dct = idct_2d(grad_dct).sign()
            mask = (grad_dct == grad.sign())
            grad *= mask
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            delta = adv_images - images

            # Tampering intensity is 8
            delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.13699, max=0.13699)
            delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.14005, max=0.14005)
            delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.13941, max=0.13941)

            # # Tampering intensity is 6
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.10274, max=0.10274)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.10504, max=0.10504)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.10456, max=0.10456)

            # # Tampering intensity is 5
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.08562, max=0.08562)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.08753, max=0.08753)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.08714, max=0.08714)

            # # Tampering intensity is 4
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.06849, max=0.06849)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.07002, max=0.07002)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.06970, max=0.06970)

            # # Tampering intensity is 3
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.05137, max=0.05137)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.05252, max=0.05252)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.05228, max=0.05228)

            # # # Tampering intensity is 2
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.03424, max=0.03424)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.03501, max=0.03501)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.03485, max=0.03485)

            # # Tampering intensity is 1
            # delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.01712, max=0.01712)
            # delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.01750, max=0.01750)
            # delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.01742, max=0.01742)

            adv_images[0][0] = torch.clamp(images[0][0] + delta[0][0], min=-2.1179, max=2.2489).detach()
            adv_images[0][1] = torch.clamp(images[0][1] + delta[0][1], min=-2.0357, max=2.4285).detach()
            adv_images[0][2] = torch.clamp(images[0][2] + delta[0][2], min=-1.8044, max=2.64).detach()

            adv_image = adv_images.squeeze(0).cpu()
            adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(adv_image[0], adv_image[1], adv_image[2],
                                                                            json_absolute_path, weights_path,
                                                                            labels.item(),
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

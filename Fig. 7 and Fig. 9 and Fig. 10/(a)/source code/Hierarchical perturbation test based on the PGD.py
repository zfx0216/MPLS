import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet
from skimage.metrics import structural_similarity as ssim
from skimage import io
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

index_file_absolute_path = "..."
weight_file_absolute_path = "..."
actual_images_folder_absolute_path = "..."

degree = 128
iteration_step_size = 0.001
ratio = 0
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

ssim_configs = [
    {"upper": 1.00000, "lower": 0.999875},
    {"upper": 0.999875, "lower": 0.999750},
    {"upper": 0.999750, "lower": 0.999625},
    {"upper": 0.999625, "lower": 0.999500},
    {"upper": 0.999500, "lower": 0.999375},
    {"upper": 0.999375, "lower": 0.999250},
    {"upper": 0.999250, "lower": 0.999125},
    {"upper": 0.999125, "lower": 0.999000}
]

with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

model = alexnet(num_classes=8).to(device)
model.load_state_dict(torch.load(weight_file_absolute_path, map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def predict_tensor(img_tensor):
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    return torch.argmax(prob, dim=1).item(), output

def calculate_gradient_tensor(img_tensor, label):
    img_tensor.requires_grad_(True)
    output = model(img_tensor)
    loss = criterion(output, torch.tensor([label], device=device))
    grad = torch.autograd.grad(loss, img_tensor)[0]
    return grad.squeeze(0).cpu().numpy()

def divide_matrix(input_matrix, level):
    threshold = np.percentile(input_matrix, level)
    return (input_matrix > threshold).astype(np.uint8)

def apply_pixel_change(img_np, grad_np, mask_np):
    delta = np.sign(grad_np) * mask_np * iteration_step_size * 255
    img_np = img_np + delta.transpose(1, 2, 0)
    return np.clip(img_np, 0, 255)

def calculate_image_ssim(img1_np, img2_np):
    return ssim(img1_np, img2_np, win_size=3)

def run_one_config(cfg, image_files):
    SSIM_UPPER = cfg["upper"]
    SSIM_LOWER = cfg["lower"]

    total_valid_images = 0
    total_label_changed = 0
    total_label_unchanged = 0
    total_iter = 0

    for image_file in image_files:
        img_path = os.path.join(actual_images_folder_absolute_path, image_file)

        img_original_np = io.imread(img_path)
        img_original_pil = Image.fromarray(img_original_np).resize((224, 224))
        img_original_np = np.array(img_original_pil)
        img_current_np = img_original_np.copy().astype(np.float64)

        img_tensor = data_transform(img_original_pil).unsqueeze(0).to(device)
        original_label, _ = predict_tensor(img_tensor)

        iter_count = 0
        stop_flag = False
        valid_images = []

        while not stop_flag and iter_count <= max_num_iterative:
            iter_count += 1
            total_iter += 1

            current_img_uint8 = img_current_np.astype(np.uint8)
            current_ssim = calculate_image_ssim(img_original_np, current_img_uint8)

            current_img_tensor = data_transform(Image.fromarray(current_img_uint8)).unsqueeze(0).to(device)
            current_label, _ = predict_tensor(current_img_tensor)
            label_changed = current_label != original_label

            if SSIM_LOWER <= current_ssim <= SSIM_UPPER:
                valid_images.append(label_changed)

            if current_ssim < SSIM_LOWER or iter_count >= max_num_iterative:
                stop_flag = True
                continue

            grad = calculate_gradient_tensor(img_tensor, original_label)
            mask = divide_matrix(np.abs(grad), ratio)
            img_current_np = apply_pixel_change(img_current_np, grad, mask)
            img_tensor = data_transform(Image.fromarray(img_current_np.astype(np.uint8))).unsqueeze(0).to(device)

        cnt = len(valid_images)
        chg = sum(valid_images)
        nchg = cnt - chg
        total_valid_images += cnt
        total_label_changed += chg
        total_label_unchanged += nchg

    success_rate = total_label_changed / total_valid_images if total_valid_images else 0.0
    fail_rate = total_label_unchanged / total_valid_images if total_valid_images else 0.0

    return {
        "upper": SSIM_UPPER,
        "lower": SSIM_LOWER,
        "valid": total_valid_images,
        "changed": total_label_changed,
        "unchanged": total_label_unchanged,
        "success_rate": success_rate,
        "fail_rate": fail_rate
    }

if __name__ == "__main__":
    image_files = [f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')]
    results = []

    for i, cfg in enumerate(ssim_configs, 1):
        print(f"\n========== Group {i} SSIM [ {cfg['lower']} ~ {cfg['upper']} ] ==========")
        res = run_one_config(cfg, image_files)
        results.append(res)
        print(f"Valid: {res['valid']} | Changed: {res['changed']} | Unchanged: {res['unchanged']}")
        print(f"Success Rate: {res['success_rate']:.2%} | Fail Rate: {res['fail_rate']:.2%}")

    print("\n" + "=" * 100)
    print("SSIM Interval Summary")
    print("=" * 100)
    print(f"{'ID':<4}{'SSIM L':<12}{'SSIM U':<12}{'Valid':<10}{'Changed':<8}{'Unchanged':<8}{'Success':<12}{'Fail':<12}")
    for i, r in enumerate(results, 1):
        print(f"{i:<4}{r['lower']:<12}{r['upper']:<12}{r['valid']:<10}{r['changed']:<8}{r['unchanged']:<8}{r['success_rate']:.2%}{'':<4}{r['fail_rate']:.2%}")
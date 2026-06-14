import torch
from torchvision import transforms
from torchvision.models import alexnet
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
import json
import os

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def sigma_zero_targeted(model,
                        inputs,
                        labels,
                        target,
                        steps: int = 300,
                        lr: float = 1.0,
                        sigma: float = 1e-3,
                        threshold: float = 0.3,
                        verbose: bool = False,
                        epsilon_budget: int = None,
                        grad_norm: float = torch.inf,
                        t: float = 0.01):

    def clamp(delta, inputs):
        channel_mins = torch.tensor([-2.1179, -2.0357, -1.8044], device=inputs.device).view(1, 3, 1, 1)
        channel_maxs = torch.tensor([2.2489, 2.4285, 2.6400], device=inputs.device).view(1, 3, 1, 1)
        return torch.max(torch.min(inputs + delta, channel_maxs), channel_mins) - inputs

    l0_approximation = lambda tensor, sigma: tensor.square().div(tensor.square().add(sigma)).sum(dim=1)

    batch_view = lambda tensor: tensor.view(tensor.shape[0], *[1] * (inputs.ndim - 1))

    normalize = lambda tensor: (
        tensor.flatten(1) / tensor.flatten(1).norm(p=grad_norm, dim=1, keepdim=True).clamp_(min=1e-12)
    ).view(tensor.shape)

    device = next(model.parameters()).device
    batch_size = inputs.shape[0]
    max_size = torch.prod(torch.tensor(inputs.shape[1:]))

    delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 20)

    best_delta = delta.clone().detach()
    best_l0 = torch.full((batch_size,), max_size.item(), device=device)
    query_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    th = torch.full_like(inputs, threshold)

    for step in range(steps):
        optimizer.zero_grad()

        active_delta = delta[query_mask].clone().detach().requires_grad_(True)
        active_inputs = inputs[query_mask]
        active_target = target[query_mask]

        adv_inputs = active_inputs + active_delta
        logits = model(adv_inputs)

        target_logit = logits.gather(1, active_target.unsqueeze(1)).squeeze(1)
        logits_masked = logits.scatter(1, active_target.unsqueeze(1), -1e9)
        max_other = logits_masked.max(1)[0][0]

        dl_loss = (max_other - target_logit).clamp(min=0)
        margin_loss = (max_other - target_logit + 10.0).clamp(min=0)
        dl_loss = margin_loss

        l0_approx = l0_approximation(active_delta.flatten(1), sigma)
        l0_approx_normalized = l0_approx / active_delta.flatten(1).shape[1]

        pred = logits.argmax(1)
        is_adv = (pred == active_target)

        true_l0 = active_delta.abs().flatten(1).gt(1e-6).sum(dim=1)
        better = (true_l0 < best_l0[query_mask]) & is_adv
        best_l0[query_mask] = torch.where(better, true_l0, best_l0[query_mask])
        best_delta[query_mask] = torch.where(batch_view(better),
                                             active_delta.detach(),
                                             best_delta[query_mask])

        total_loss = (1.0 - is_adv.float()).mean() + dl_loss.mean() + 5.0 * l0_approx_normalized.mean()

        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"Step {step:3d} | Loss: {total_loss.item():.4f} | "
                  f"L0 median: {true_l0.median().item():.1f} | "
                  f"Success rate: {is_adv.float().mean().item():.3f} | "
                  f"Best L0: {best_l0[query_mask].float().mean().item():.1f}")

        total_loss.backward()

        delta.grad = torch.zeros_like(delta)
        delta.grad[query_mask] = active_delta.grad

        delta.grad.data = normalize(delta.grad.data)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            delta.data = clamp(delta.data, inputs)

            th_active = th[query_mask]
            th_active[is_adv] += t * scheduler.get_last_lr()[0]
            th_active[~is_adv] -= t * scheduler.get_last_lr()[0]
            th[query_mask] = th_active.clamp(0, 1)

            delta.data[delta.data.abs() < th] = 0

            if epsilon_budget is not None:
                query_mask[best_l0 <= epsilon_budget] = False

            if not query_mask.any():
                break

    return inputs + best_delta

# Calculate targeted attack categories
def target_index_image_path(image_path, index_path, weight_path, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = preprocess(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model_current
    model = model_cnn(num_classes=1000).to(device)
    # Load model_current weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model_current to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(classification_probability, 1000)
    attack_index = top_indices[499]
    return(attack_index)


def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = preprocess(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model_current
    model = model_cnn(num_classes=1000).to(device)
    # Load model_current weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model_current to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()
    return(predicted_class_index)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index_file_absolute_path = r"..."

    weight_file_absolute_path = r"..."

    actual_images_folder_absolute_path = "..."
    output_directory = "..."

    # The total number of images in the folder that require tampering attacks
    the_total_number_of_tampered_images = 0

    image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')],
                         key=sort_func)

    # Record the total number of original images
    image_num = 0

    # Record the total number of successfully attacked images
    success_num = 0

    for image_file in image_files:
        print(f"The image number currently being processed is: {image_file}")
        image_num = image_num + 1

        the_total_number_of_tampered_images = the_total_number_of_tampered_images + 1

        actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

        model_current = alexnet

        model = alexnet(weights=None)
        model.load_state_dict(torch.load(weight_file_absolute_path))
        model.eval()
        model.to(device)

        img = Image.open(actual_image_absolute_path).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        x = preprocess(img).unsqueeze(0).to(device)

        true_label = predict_image_path(actual_image_absolute_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)

        target_class = target_index_image_path(actual_image_absolute_path, index_file_absolute_path, weight_file_absolute_path, model_current)             # 目标：tench（随便改成 1~999 任意类）

        true_tensor = torch.tensor([true_label], device=device)
        target_tensor = torch.tensor([target_class], device=device)

        adv_x = sigma_zero_targeted(
            model=model,
            inputs=x,
            labels=true_tensor,
            target=target_tensor,
            steps=400,
            lr=1.0,
            threshold=0.3,
            verbose=True,
            epsilon_budget=None,
        )

        adv_img = adv_x.squeeze(0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        adv_img = adv_img * std + mean
        adv_img = torch.clamp(adv_img, 0, 1)

        adv_image_path = f"{output_directory}\\{image_num}.png"
        transforms.ToPILImage()(adv_img).save(adv_image_path)
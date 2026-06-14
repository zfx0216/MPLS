from torchvision.models import alexnet
import torch.optim.lr_scheduler as lr_scheduler
from adv_lib.utils.losses import difference_of_logits
import os
import json
import torch
from PIL import Image
from torchvision import transforms


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def sigma_zero(model,
               inputs,
               labels,
               steps: int = 1000,
               lr: float = 1.0,
               sigma: float = 1e-3,
               threshold: float = 0.3,
               verbose: bool = False,
               epsilon_budget=None,
               grad_norm=torch.inf,
               t=0.01
               ):

    def clamp(delta, inputs):
        channel_mins = torch.tensor([-2.1179, -2.0357, -1.8044], device=inputs.device).view(1, 3, 1, 1)
        channel_maxs = torch.tensor([2.2489, 2.4285, 2.6400], device=inputs.device).view(1, 3, 1, 1)

        return torch.max(torch.min(inputs + delta, channel_maxs), channel_mins) - inputs

    l0_approximation = lambda tensor, sigma: tensor.square().div(tensor.square().add(sigma)).sum(dim=1)

    batch_view = lambda tensor: tensor.view(tensor.shape[0], *[1] * (inputs.ndim - 1))

    normalize = lambda tensor: (
            tensor.flatten(1) / tensor.flatten(1).norm(p=grad_norm, dim=1, keepdim=True).clamp_(min=1e-12)).view(
        tensor.shape)

    device = next(model.parameters()).device
    batch_size, max_size = inputs.shape[0], torch.prod(torch.tensor(inputs.shape[1:]))

    delta = torch.zeros_like(inputs, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 10)
    best_delta = delta.clone()
    query_mask = torch.full((batch_size,), True, device=device)
    best_l0 = torch.full((batch_size,), max_size, device=device)
    is_adv_below_eps = torch.full((batch_size,), False, device=device)
    th = torch.ones(size=inputs.shape, device=device) * threshold

    for i in range(steps):
        optimizer.zero_grad()

        active_delta = delta[query_mask].clone().detach().requires_grad_(True)
        active_inputs = inputs[query_mask]
        active_labels = labels[query_mask]

        adv_inputs = active_inputs + active_delta

        # compute loss
        logits = model(adv_inputs)
        dl_loss = difference_of_logits(logits, active_labels).clip(0)
        l0_approx = l0_approximation(active_delta.flatten(1), sigma)
        l0_approx_normalized = l0_approx / active_delta.data.flatten(1).shape[1]

        # keep best solutions
        predicted_classes = (logits).argmax(1)
        true_l0 = active_delta.data.flatten(1).ne(0).sum(dim=1)
        is_not_adv = predicted_classes == active_labels
        is_smaller = true_l0 < best_l0[query_mask]
        is_both = ~is_not_adv & is_smaller
        best_l0[query_mask] = torch.where(is_both, true_l0.detach(), best_l0[query_mask])
        best_delta[query_mask] = torch.where(batch_view(is_both), active_delta.data.clone().detach(),
                                             best_delta[query_mask])
        is_adv_below_eps = best_l0 <= epsilon_budget if epsilon_budget is not None else is_adv_below_eps

        # update step
        adv_loss = (is_not_adv + dl_loss + l0_approx_normalized).mean()

        if verbose and i % 100 == 0:
            print(th.flatten(1).mean(dim=1), th.flatten(1).mean(dim=1).shape)
            print(is_not_adv)
            print(
                f"iter: {i}, dl loss: {dl_loss.mean().item():.4f}, l0 normalized loss: {l0_approx_normalized.mean().item():.4f}, current median norm: {delta.data.flatten(1).ne(0).sum(dim=1).median()}")

        adv_loss.backward()

        if delta.grad is None:
            delta.grad = torch.zeros_like(delta, device=device)
        # Copy gradients from active_delta.grad to delta.grad at the masked positions
        delta.grad[query_mask] += active_delta.grad
        delta.grad.data = normalize(delta.grad.data)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # enforce box constraints
            delta.data = clamp(delta.data, inputs)
            # dynamic thresholding step
            th_active = th[query_mask]
            th_active[is_not_adv, :, :, :] -= t * scheduler.get_last_lr()[0]
            th_active[~is_not_adv, :, :, :] += t * scheduler.get_last_lr()[0]
            th[query_mask] = th_active
            th.clamp_(0, 1)
            # filter components
            delta.data[delta.data.abs() < th] = 0
            # update active set
            query_mask[is_adv_below_eps] = False
            if not any(query_mask):
                break
    return (inputs + best_delta)


def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = preprocess(img)
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
    return(predicted_class_index)



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index_file_absolute_path = r"..."

    weight_file_absolute_path = r"..."

    actual_images_folder_absolute_path = "..."
    output_directory = "..."
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

        preprocess = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        model = alexnet(weights=None)
        model.load_state_dict(torch.load(weight_file_absolute_path))
        model.eval()
        model.to(device)

        img = Image.open(actual_image_absolute_path).convert("RGB")

        original_class = predict_image_path(actual_image_absolute_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)

        x = preprocess(img).unsqueeze(0).to(device)

        adv = sigma_zero(
            model,
            x,
            labels=torch.tensor([original_class], device=device),
            steps=200,
            lr=1.0,
            threshold=0.3,
            verbose=True
        )

        adv_img = adv.squeeze().detach().cpu()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        adv_img = adv_img * std + mean
        adv_img = torch.clamp(adv_img, 0.0, 1.0)
        adv_pil = transforms.ToPILImage()(adv_img)
        adv_image_path = f"{output_directory}\\{image_num}.png"
        adv_pil.save(adv_image_path)
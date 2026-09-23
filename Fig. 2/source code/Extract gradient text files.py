import torch
import numpy as np
import os
from PIL import Image
from torchvision import models, transforms

IMAGE_PATH = r"..."
MODEL_WEIGHTS_PATH = r"..."
SAVE_DIR = r"..."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.alexnet(weights=None)
model.classifier[6] = torch.nn.Linear(4096, 50)

model.load_state_dict(
    torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
)
model = model.to(DEVICE)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(image).unsqueeze(0).to(DEVICE)
img_tensor.requires_grad = True

output = model(img_tensor)
target_class = torch.argmax(output).item()
output[0, target_class].backward()

grad = img_tensor.grad.squeeze().detach().cpu().numpy()
R_weights = grad[0]
G_weights = grad[1]
B_weights = grad[2]

np.savetxt(os.path.join(SAVE_DIR, "pixel_weight_matrix_R.txt"), R_weights, fmt="%.10f")
np.savetxt(os.path.join(SAVE_DIR, "pixel_weight_matrix_G.txt"), G_weights, fmt="%.10f")
np.savetxt(os.path.join(SAVE_DIR, "pixel_weight_matrix_B.txt"), B_weights, fmt="%.10f")
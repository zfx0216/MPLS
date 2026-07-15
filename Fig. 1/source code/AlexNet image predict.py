"""
The main function of this code file is to use the officially trained AlexNet model to predict the category of a single image.
First, you need to set "img_absolute_path" to the absolute path of the image to be predicted.
Second, you need to set "json_absolute_path" to the absolute path of the index file containing 1000 categories. This index file is named "output_class_indices.json" in the current directory.
Third, you need to set "weights_absolute_path" to the weights file of the officially trained AlexNet model, the download link of which is provided in the paper.
Finally, executing this code file will output the prediction category information for this image.
"""


import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models import alexnet


def main():
    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Perform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Load image
    img_absolute_path = "..."
    assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
    img = Image.open(img_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Read class_indict
    json_absolute_path = "..."
    assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

    with open(json_absolute_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = alexnet(num_classes=1000).to(device)

    # Load model weights
    weights_absolute_path = "..."
    assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
    model.load_state_dict(torch.load(weights_absolute_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top_probs, top_indices = torch.topk(predict, 1000)

    for i in range(5):
        class_index = top_indices[i].item()
        class_prob = top_probs[i].item()
        print("Top {}: index: {}  class: {:10}   Classification probability: {:.5f}   Classification value: {:.6f}".format(i + 1,class_index, class_indict[str(class_index)],
                                                                              class_prob, output[class_index].numpy()))

    # # Get specified category information,Specifically, set which category in "[]" "()"to output
    # print("class: {:10}   Classification probability: {:.5f}   Classification value: {:.5f}".format(class_indict[str(500)], predict[500].numpy(),
    #                                                                   output[500].numpy()))

    plt.show()


if __name__ == '__main__':
    main()

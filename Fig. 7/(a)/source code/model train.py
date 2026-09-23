"""
The current code file is used for training convolutional neural network models. You only need to set the parameter "root" to the absolute path of the training set
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import alexnet
import matplotlib.pyplot as plt


def main():
    loss_sequence = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    train_dataset = datasets.ImageFolder(
        root=r"...",
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    print(cla_dict)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json.dumps(cla_dict, indent=4))

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )

    net = alexnet(num_classes=8)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    epochs = 100
    best_loss = float('inf')
    best_model_path = 'AlexNetbest.pth'

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} lr:{:.6f}".format(
                epoch + 1, epochs, loss, optimizer.param_groups[0]['lr']
            )

        epoch_loss = running_loss / len(train_loader)
        loss_sequence.append(epoch_loss)
        print('[epoch %d] train_loss: %.3f  lr: %.6f' %
              (epoch + 1, epoch_loss, optimizer.param_groups[0]['lr']))

        scheduler_step.step()
        scheduler_plateau.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(net.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")

        plt.clf()
        plt.plot(range(len(loss_sequence)), loss_sequence, marker="o")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig('loss_plot.png')
        plt.show()

    print('Finished Training')
    del net


if __name__ == '__main__':
    main()




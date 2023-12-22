import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_planes():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    images, labels = next(iter(train_dataloader))

    new_images = images[labels==0]
    new_labels = labels[labels==0]

    VALIDATE = False
    if VALIDATE:
        print(new_images.shape)
        print(new_labels.shape)

        for i in range(5):
            an_image = new_images[i*10]
            an_image = an_image.permute(1, 2, 0).cpu().numpy()
            plt.imshow(an_image)
            plt.show()

    return new_images
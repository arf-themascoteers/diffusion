import matplotlib.pyplot as plt
import torch


def view_it(image=None):
    if image is None:
        loaded_tensor = torch.load('saved_tensor.pt')
        image = loaded_tensor[-1]
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        return out


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = MLP(input_size=784, hidden_size=256, output_size=10).to(device)

model.load_state_dict(torch.load("model.pth"))
model.eval()


##Lets visualize the what the digit thinks
## For more realistic visualizations lets use frequency loss + transformations
def pixel_frequency_loss(x):
    """
    Calculates the pixel frequency loss for a 3D torch tensor representing an image.

    The loss encourages nearby pixels to have similar values, reducing the variance of
    nearby pixels.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, height, width].

    Returns:
        torch.Tensor: Scalar value representing the pixel frequency loss.
    """
    batch_size, height, width = x.shape

    # Calculate the difference between each pixel and its neighbors
    dx = x[:, 1:, :] - x[:, :-1, :]
    dy = x[:, :, 1:] - x[:, :, :-1]

    # Calculate the variance of the differences
    pixel_freq_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

    return pixel_freq_loss



def main(target_number):
    target = torch.tensor([target_number]).to(device)
    prob_loss_fn = nn.CrossEntropyLoss()
    transformation_eval = transforms.Compose(
    [
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    )   
    transformation = transforms.Compose(
        [
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(40, (0.01, 0.2)),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    dataset = MNIST(root="data/", download=True, transform=transforms.ToTensor())
    for i in range(len(dataset)):
        if dataset[i][1] == target_number:
            image = dataset[i][0]
            break
    
    for i in range(10):
        trans_image = transformation(image).to(device)
        trans_image.requires_grad = True
        trans_image_reshape = trans_image.view(-1, 28 * 28)
        pred = model(trans_image_reshape)
        prob_loss = prob_loss_fn(pred, target)
        prob_loss.backward()

        plt.imshow(image.squeeze(0).cpu().detach().numpy())
        plt.axis("off")
        # plt.savefig(f"{target_number}_orig_visualization.png")
        plt.show()
        plt.imshow(trans_image.grad.squeeze(0).cpu().detach().numpy())
        plt.axis("off")
        # plt.savefig(f"{target_number}_grad_visualization.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize What the model looks at in an image')

    # Add the integer argument
    parser.add_argument('--targetnumber', type=int, required=True, help='Target Number to Visualize(0-9)')

    # Parse the arguments
    args = parser.parse_args()

    main(args.targetnumber)

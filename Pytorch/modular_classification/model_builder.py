"""
Module that contains the architecture of the TinnyVGG
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    Create the TinnyVGG architecture.
    Args:
    -----
        input_shape: An integer that represents the number of input neurons.
        hidden_units: The number of neurons in the hidden layers
        output_shape: The number of output neurons, its depends of the classes
    """
    def __init__(self, input_shape: int, hidden_units: int,
                 output_shape: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),
                      # options = "valid" (no padding) or
                      # "same" (output has same shape as input)
                      # or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses
            # and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        """Function that arranges the model architecture"""
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        # Decoder
        x = F.relu(self.dec_conv1(x))
        x = torch.sigmoid(self.dec_conv2(x))
        return x

class AutoV2(nn.Module):
    def __init__(self, image_size=(3,512,512)):
        super(AutoV2, self).__init__()
        
        # Assume image_size is a tuple (C, H, W)
        _, H, W = image_size
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = self._calculate_conv_output_size(H, W)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * conv_output_size * conv_output_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)

        # Decoder
        self.dec_fc1 = nn.Linear(in_features=32, out_features=64)
        self.dec_fc2 = nn.Linear(in_features=64, out_features=128)
        self.dec_fc3 = nn.Linear(in_features=128, out_features=64 * conv_output_size * conv_output_size)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        encoded = F.relu(self.fc3(x))

        # Decoder
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = x.view(x.size(0), 64, self.conv_output_size, self.conv_output_size)  # Reshape
        x = F.relu(self.dec_conv1(x))
        x = F.sigmoid(self.dec_conv2(x))
        return x

    def _calculate_conv_output_size(self, H, W):
        # Apply the formula for each conv layer to calculate output size
        def conv_output(size, kernel_size=3, stride=2, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        H, W = conv_output(H), conv_output(W)  # After first conv layer
        H, W = conv_output(H), conv_output(W)  # After second conv layer

        self.conv_output_size = H  # Assuming H and W are equal
        return self.conv_output_size
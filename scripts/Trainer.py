from data import PaintsTorchDataset
from model import Autoencoder, AutoV2, AutoV2_Lite, Autoencoder_Max, Autoencoder_3, Autoencoder_3_Ultimate

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from torchvision.utils import make_grid
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from pathlib import Path

from torchvision.utils import save_image

def save_images(epoch, model, dataset, device):
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    lineart, palette, illustration = sample.lineart.to(device), sample.palette.to(device), sample.illustration.to(device)
    input = torch.cat((lineart, palette), dim=0).unsqueeze(0)
    output = model(input)

    # Normalize tensors to be in the range [0, 1] for image saving
    lineart_img = (lineart + 1) / 2
    palette_img = (palette + 1) / 2
    output_img = (output.squeeze(0) + 1) / 2
    illustration_img = (illustration + 1) / 2
    #print(lineart_img.shape, palette_img.shape, output_img.shape, illustration_img.shape)
    lineart_img = lineart_img.repeat(3, 1, 1)  # Repeat the single channel to create a 3-channel image

    # Concatenate images side by side
    all_images = torch.cat([lineart_img, palette_img, output_img, illustration_img], dim=2)
    # Convert to PIL Image for annotation
    all_images_pil = transforms.ToPILImage()(all_images.cpu())
    draw = ImageDraw.Draw(all_images_pil)
    font = ImageFont.truetype("arial.ttf", 20)

    section_width = all_images.shape[2] // 4
    labels = ["Lineart", "Palette", "Output", "Illustration"]
    for i, label in enumerate(labels):
        draw.text((i * section_width + 10, 10), label, (255, 255, 255), font=font)

    # Save the annotated image
    all_images_pil.save(f"epochsTestifier/epoch_{epoch}.png")

def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in tqdm(test_loader, desc=f"Evaluating...", leave=False):
            lineart, palette, illustration = data.lineart, data.palette, data.illustration
            lineart, palette, illustration = lineart.to(device), palette.to(device), illustration.to(device)

            inputs = torch.cat((lineart, palette), dim=1)
            outputs = model(inputs)

            loss = criterion(outputs, illustration)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss

# Function for training loop
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, dataset):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            lineart, palette, illustration = data.lineart, data.palette, data.illustration
            lineart, palette, illustration = lineart.to(device), palette.to(device), illustration.to(device)

            inputs = torch.cat((lineart, palette), dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, illustration)
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_loss:.4f}")
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss after Epoch {epoch+1}: {test_loss:.4f}")
        # Save Model
        torch.save(model.state_dict(), os.path.join("D:\Dataset\PaintTorch\models", f"{epoch+1}--Autoencoder_3_Ultimate--{epoch_loss:.4f}.pth"))

        # Save images at the end of each epoch
        save_images(epoch + 1, model, dataset, device)

if __name__ == "__main__":
    # Parameters and Data Loader Setup
    train_dataset = PaintsTorchDataset( Path("D:/Dataset/PaintTorch/prepared/prepared/train"))
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = PaintsTorchDataset( Path("D:/Dataset/PaintTorch/prepared/prepared/test"))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Device and Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder_3_Ultimate().to(device).to(torch.bfloat16)

    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Number of Training Epochs
    num_epochs = 10000

     # Start Training
    train(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device, test_dataset)
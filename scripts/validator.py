from data import PaintsTorchDataset
from model import Autoencoder, AutoV2, AutoV2_Lite, Autoencoder_Max, Autoencoder_3, Autoencoder_3_Ultimate

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import glob
from pathlib import Path
import csv

csv_file = "summary.csv"

models_directory = 'D:/GithUB/AI_CG_colors/models_checkpoints/auto3_ultimate'
model = Autoencoder_3_Ultimate()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).to(torch.bfloat16)

dataset = PaintsTorchDataset( Path("D:/Dataset/PaintTorch/prepared/prepared/val"))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def process_model(model,file):
    model.eval() 
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in tqdm(dataloader, desc=f"Evaluating {file} ...", leave=False):
            lineart, palette, illustration = data.lineart, data.palette, data.illustration
            lineart, palette, illustration = lineart.to(device), palette.to(device), illustration.to(device)

            inputs = torch.cat((lineart, palette), dim=1)
            outputs = model(inputs)

            loss = criterion(outputs, illustration)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    #print(f"{model_file} --> {avg_loss}")
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_file, avg_loss])
    pass

for model_file in tqdm(glob.glob(os.path.join(models_directory, '*.pth')), desc="Processing models"):
    #print(f"Processing {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model_filename = os.path.basename(model_file)
    process_model(model,model_filename)
from data import PaintsTorchDataset

from torch.utils.data import DataLoader

dataset = PaintsTorchDataset(path_to_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

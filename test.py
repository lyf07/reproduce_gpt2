import torch
from torch.utils.data import DataLoader
import os

import model
from model import *
import train
from train import *
import prepare_data

import warnings
warnings.filterwarnings("ignore")

checkpoint_dir = 'checkpoints'
model_path = os.path.join(os.getcwd(), checkpoint_dir, 'best_model.pt')

if __name__ == '__main__':
    if not os.path.exists(model_path):
        print("Model not found!")
        exit()
    checkpoint = torch.load(model_path)
    train_config = checkpoint['train_config']
    model_config = checkpoint['model_config']
    
    device = train_config.device
    step = checkpoint["step"]
    best_loss = checkpoint["loss"]
    print(f"best_loss = {best_loss}")
    gpt = model.GPT(model_config)
    gpt.load_state_dict(checkpoint['model'])
    train_data, test_data = prepare_data.get_data()
    test_dataloader  = DataLoader(prepare_data.MyDataset(test_data, train_config), batch_size=train_config.batch_size, shuffle=True)
    
    tot_loss = 0
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            response, loss = gpt(data, target)
            tot_loss += loss.detach()
    print(tot_loss / len(test_dataloader))
    

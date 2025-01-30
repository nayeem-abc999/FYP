
#Jaccard: 0.2150 - F1: 0.3144 - Recall: 0.2783 - Precision: 0.6029 - Acc: 0.9946 512/2100 -35 epochs - 0.0001 LR
#Jaccard: 0.2266 - F1: 0.3258 - Recall: 0.3082 - Precision: 0.5341 - Acc: 0.9945 512/2100 -45 epochs - 0.0001 LR
#Jaccard: 0.2174 - F1: 0.3201 - Recall: 0.2985 - Precision: 0.5670 - Acc: 0.9943 512/2100 -45 epochs - 0.001 LR
#Jaccard: 0.2067 - F1: 0.3078 - Recall: 0.2839 - Precision: 0.5390 - Acc: 0.9947 512/2100 -45 epochs - 0.01 LR
#Jaccard: 0.2118 - F1: 0.3140 - Recall: 0.2994 - Precision: 0.5412 - Acc: 0.9945 512/2100 - 70 epochs - 0.001 LR
#Jaccard: 0.2129 - F1: 0.3155 - Recall: 0.2882 - Precision: 0.5512 - Acc: 0.9945 800/2100 - 70 epochs - 0.001 LR
#Jaccard: 0.2446 - F1: 0.3553 - Recall: 0.3265 - Precision: 0.5862 - Acc: 0.9947 800/2100 - 70 epochs - 0.0001 LR


# UNET attention
# Jaccard: 0.2386 - F1: 0.3456 - Recall: 0.2991 - Precision: 0.6012 - Acc: 0.9946


# UNET with IDRirD white pngs
# Jaccard: 0.3381 - F1: 0.4803 - Recall: 0.4562 - Precision: 0.5651 - Acc: 0.9890 25 epochs/ 0.0001 LR
# Jaccard: 0.3080 - F1: 0.4433 - Recall: 0.4214 - Precision: 0.5634 - Acc: 0.9896 35 epochs/ 0.0001 LR

import time
from glob import glob
import torch
from torch.utils.data import DataLoader

from data import RetinalDataset
from model import UNet
from loss import DiceBCELoss
from utils import seeding, create_directory, epoch_time

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    # Seeding 
    seeding(42)

    # Directories
    create_directory("files")

    # Load dataset
    train_x = sorted(glob("./final_dataset/train/images/*"))
    train_y = sorted(glob("./final_dataset/train/masks/*"))

    valid_x = sorted(glob("./final_dataset/test/images/*"))
    valid_y = sorted(glob("./final_dataset/test/masks/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    # Hyperparameters
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 2
    lr = 0.0001
    checkpoint_path = "files/checkpoint.pth"

    # Dataset and loader
    train_dataset = RetinalDataset(train_x, train_y)
    valid_dataset = RetinalDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = UNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint."
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

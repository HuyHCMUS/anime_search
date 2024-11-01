# main.py
import argparse
import torch
from dataset import AnimeDataset, transform_train, transform_val
from train_eval import train_one_epoch, validate
from utils import save_checkpoint, load_checkpoint
from ViT_model import ViTModel
from Facenet_model import FaceNetModel
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import os

def main(args):
    # Load data
    full_dataset = AnimeDataset(args.data_path, transform=transform_train) 
    train_size = int(0.8 * len(full_dataset))  # 80% cho train
    val_size = len(full_dataset) - train_size  # 20% cho val
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    if args.model == 'vit':
        model = ViTModel(num_classes=args.num_classes)
    elif args.model == 'facenet':
        model = FaceNetModel(num_classes=args.num_classes)
    
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    
    # Load checkpoint if resume is True
    if args.resume and os.path.isfile(args.checkpoint_path):
        print(f"Loading checkpoint from '{args.checkpoint_path}'")
        checkpoint = load_checkpoint(model, optimizer, filename=args.checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch
        scheduler.step(start_epoch)  # Adjust scheduler to current epoch
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_top5_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, val_top5_acc = validate(model, val_loader, criterion, args.device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%,"
              f"Train top5 Acc: {train_top5_acc:.2f}%, Val top5 Acc: {val_top5_acc:.2f}%")
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, filename=f"checkpoint_epoch{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['vit', 'facenet'], default='vit')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=4000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint if available")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help="Path to checkpoint file")
    args = parser.parse_args()
    
    main(args)

# train_eval.py
import torch
from utils import calculate_accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    for images, characters in train_loader:
        images = images.to(device)
        labels = characters.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc1, acc5 = calculate_accuracy(outputs, labels, top_k=(1, 5))
        correct += acc1.item()
        top5_correct += acc5.item()
        total += labels.size(0)

    return running_loss / len(train_loader), correct / total * 100, top5_correct / total * 100

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for images, characters in val_loader:
            images = images.to(device)
            labels = characters.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            acc1, acc5 = calculate_accuracy(outputs, labels, top_k=(1, 5))
            correct += acc1.item()
            top5_correct += acc5.item()
            total += labels.size(0)

    return val_loss / len(val_loader), correct / total * 100, top5_correct / total * 100

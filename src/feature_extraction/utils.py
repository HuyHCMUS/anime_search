import torch
# import torchvision.transforms as transforms
# from PIL import Image

def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': model.epoch,  # thêm số epoch vào checkpoint
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def calculate_accuracy(outputs, labels, top_k=(1,)):
    max_k = max(top_k)
    batch_size = labels.size(0)
    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





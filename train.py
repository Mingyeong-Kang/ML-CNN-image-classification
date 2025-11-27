import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_cifar10_loaders
from models.baseline_cnn import BaselineCNN
from models.resnet18_finetune import ResNet18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "resnet18"])
    parser.add_argument("--epochs", type=int, default=2)  # 오늘은 1~2 epoch만
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def get_model(name: str):
    if name == "baseline":
        return BaselineCNN(num_classes=10)
    elif name == "resnet18":
        return ResNet18(num_classes=10, pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    model = get_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}] - Model: {args.model}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"|| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    import json
    filename = f"results_{args.model}.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nSaved training history to {filename}")

if __name__ == "__main__":
    main()
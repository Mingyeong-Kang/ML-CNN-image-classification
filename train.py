import argparse
import json  # 결과 JSON으로 저장
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_cifar10_loaders
from models.baseline_cnn import BaselineCNN
from models.resnet18_finetune import ResNet18
from models.efficientnet import get_efficientnet

# y_true/y_pred + Grad-CAM에서에 쓰기 위한 CIFAR-10 클래스 이름들
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "resnet18", "efficientnet"],)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def get_model(name: str):
    if name == "baseline":
        return BaselineCNN(num_classes=10)
    elif name == "resnet18":
        return ResNet18(num_classes=10, pretrained=True)
    elif name == "efficientnet":
        return get_efficientnet(num_classes=10, pretrained=True)
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

# 테스트 데이터셋에 대한 평가수행 함수
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 전체 정답 레이블과 예측값을 모아둘 바구니 2개
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            # CPU로 가져와 리스트에 계속 추가
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    # loss, acc + 전체 y_true / y_pred 리턴
    return epoch_loss, epoch_acc, all_targets, all_preds


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
    "test_acc": [],
    # confusion matrix, class별 정확도용 키
    "y_true": [],  # 마지막 epoch에서의 전체 정답 라벨
    "y_pred": [],  # 마지막 epoch에서의 전체 예측 라벨
    "class_names": CLASS_NAMES,  # CIFAR-10 클래스 이름
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}] - Model: {args.model}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # evaluate return
        test_loss, test_acc, y_true, y_pred = evaluate(
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

        # 매 epoch마다 마지막 평가 결과로 y_true / y_pred를 덮어씀 (최종 모델의 예측)
        history["y_true"] = y_true
        history["y_pred"] = y_pred

    # resnet18 학습이 끝났을 때 weight를 저장 (Grad-CAM)
    if args.model == "resnet18":
        torch.save(model.state_dict(), "resnet18_cifar10.pth")
        print("Saved ResNet18 weights to resnet18_cifar10.pth")

    filename = f"results_{args.model}.json"
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nSaved training history to {filename}")

if __name__ == "__main__":
    main()
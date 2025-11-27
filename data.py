import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    CIFAR-10 train/test DataLoader 반환.
    처음 실행 시 ./data 폴더에 자동 다운로드됨.
    """
    # 데이터 증강 + 정규화 (train용)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # test용: 증강 X, 정규화만
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,          # 여기서 자동 다운로드
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,          # 이미 있으면 다시 안 받음
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
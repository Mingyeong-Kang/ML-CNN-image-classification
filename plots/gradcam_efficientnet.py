# Grad-CAM으로 ResNet18이 CIFAR-10 이미지를 볼 때 어디를 보고 있는지 heatmap을 만들어준다.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트를 모듈 검색 경로에 추가(get_cifar10_loaders 로드 위함)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import get_cifar10_loaders
from models.efficientnet import get_efficientnet


# CIFAR-10 클래스 이름
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class GradCAM:
    # 마지막 conv layer의 feature map + gradient를 이용해 Grad-CAM heatmap을 만드는 클래스
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # forward hook: feature map 저장
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        # backward hook: gradient 저장
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # output shape: (B, C, H, W)
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] shape: (B, C, H, W)
        self.gradients = grad_output[0].detach()

    def generate(self, scores, class_idx):
        # scores: model output (logits), shape (1, num_classes)
        # class_idx: heatmap을 만들고 싶은 클래스 인덱스

        # 해당 클래스 score만 backward
        self.model.zero_grad()
        target = scores[0, class_idx]
        target.backward(retain_graph=True)

        # B=1이므로 index 0만 사용
        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # 채널별 평균 gradient = 채널 중요도
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # 가중합으로 heatmap 생성
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()  # (H, W) in [0,1]

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def overlay_heatmap_on_image(img, cam, alpha=0.4):
    # img: (3, H, W) tensor, [0,1] 범위라고 가정
    # cam: (H, W) numpy array, [0,1] 범위
    # → 원본 이미지 위에 heatmap을 얹은 RGB numpy array 반환 (H, W, 3)
    img_np = img.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)

    # matplotlib colormap으로 heatmap을 RGB로 변환
    heatmap = plt.get_cmap("jet")(cam)[:, :, :3]  # (H, W, 3), alpha 채널 제거

    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay

# EfficientNet 안에서 "마지막 Conv2d 레이어"를 자동으로 찾아주는 함수
import torch.nn as nn
def find_last_conv_layer(model):
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the model for Grad-CAM.")
    return last_conv

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for Grad-CAM: {device}")

    # 1) CIFAR-10 test loader에서 한 배치만 가져오기
    _, test_loader = get_cifar10_loaders(batch_size=16)

    dataset = test_loader.dataset   # test_loader 뒤에 숨은 실제 Dataset 객체
    n = len(dataset)

    k = 4   # Grad-CAM으로 보고 싶은 이미지 개수
    idxs = torch.randperm(n)[:k]   # 0 ~ n-1 사이에서 인덱스 k개 랜덤 뽑기

    imgs = []
    lbls = []
    for idx in idxs:
        img, lbl = dataset[idx.item()]   # dataset에서 직접 꺼내오기 (transform까지 적용된 상태)
        imgs.append(img)
        lbls.append(lbl)

    images = torch.stack(imgs).to(device)   # (k, 3, 32, 32) 텐서로 합치고, 라벨도 텐서로 변환
    labels = torch.tensor(lbls, device=device)
    num_samples = images.size(0)


    # 2) 학습된 EfficientNet 모델 로드  # CHANGED
    model = get_efficientnet(num_classes=10, pretrained=False).to(device)  # CHANGED
    state_dict = torch.load("efficientnet_cifar10.pth", map_location=device)  # CHANGED
    model.load_state_dict(state_dict)
    model.eval()


    # 3) Grad-CAM 대상 레이어 지정 (EfficientNet의 마지막 Conv 레이어 자동 탐색)  # CHANGED
    target_layer = find_last_conv_layer(model)  # CHANGED
    gradcam = GradCAM(model, target_layer)


    # 4) 몇 장만 선택해서 Grad-CAM 시각화 (예: 4장)
    num_samples = 4
    num_samples = min(num_samples, images.size(0))

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    # 한 번에 전체 forward (softmax는 보기용)
    with torch.no_grad():
        outputs = model(images)  # (B, num_classes)
        probs = F.softmax(outputs, dim=1)

    for i in range(num_samples):
        img = images[i]
        label = labels[i].item()
        pred = probs[i].argmax().item()


        # 5) 이 샘플에 대해 Grad-CAM heatmap 생성
        # backward를 위해 다시 single-sample forward
        scores = model(img.unsqueeze(0))
        cam = gradcam.generate(scores, class_idx=pred)
        overlay = overlay_heatmap_on_image(img, cam)

        # 왼쪽: 원본
        axes[i, 0].imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        axes[i, 0].set_title(f"Label: {CLASS_NAMES[label]}\nPred: {CLASS_NAMES[pred]}")
        axes[i, 0].axis("off")

        # 오른쪽: Grad-CAM overlay
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("plots/gradcam_efficientnet_examples.png")  # CHANGED
    print("Saved Grad-CAM examples to plots/gradcam_efficientnet_examples.png")

    gradcam.close()


if __name__ == "__main__":
    main()

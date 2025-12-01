import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_cm(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        annot=False
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    # 성능 좋은 모델(resnet18)의 결과를 사용
    results = load_results("results_resnet18.json")  # 파일은 프로젝트 루트에 있음
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    class_names = results["class_names"]

    plot_cm(y_true, y_pred, class_names, "plots/resnet18_confusion_matrix.png")

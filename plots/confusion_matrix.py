import json
import os  # for checking multiple possible result file paths
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse  # to parse --model argument

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

# 모델 이름으로 적절한 results_*.json 파일을 찾아주는 헬퍼 함수
def load_results_for_model(model_name: str):
    """
    Try to load results for given model name from either:
      - results/results_<model>.json
      - results_<model>.json
    """
    candidates = [
        os.path.join("results", f"results_{model_name}.json"),
        f"results_{model_name}.json",
    ]

    for path in candidates:
        if os.path.exists(path):
            return load_results(path)

    raise FileNotFoundError(
        f"Results file for model '{model_name}' not found. Tried: {', '.join(candidates)}"
    )

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
    parser = argparse.ArgumentParser()  # 모델명을 인자로 받기 위해 추가
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["baseline", "resnet18", "efficientnet"],
        help="Which model results to use for the confusion matrix.",
    )
    args = parser.parse_args()

    # # 성능 좋은 모델(resnet18)의 결과를 사용
    # results = load_results("results_resnet18.json")  # 파일은 프로젝트 루트에 있음
    # y_true = results["y_true"]
    # y_pred = results["y_pred"]
    # class_names = results["class_names"]

    # plot_cm(y_true, y_pred, class_names, "plots/resnet18_confusion_matrix.png")

    # 하드코딩된 resnet18 대신, 인자로 받은 모델의 결과를 사용
    results = load_results_for_model(args.model)
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    class_names = results["class_names"]

    save_path = os.path.join("plots", f"{args.model}_confusion_matrix.png")  # CHANGED
    plot_cm(y_true, y_pred, class_names, save_path)
    print(f"✅ Saved confusion matrix to {save_path}")

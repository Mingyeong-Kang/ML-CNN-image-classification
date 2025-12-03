import json
import os  # NEW: 결과 파일 경로 확인용
import numpy as np
import matplotlib.pyplot as plt
import argparse  # NEW: --model 인자 파싱용

def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accs = []
    for i in range(len(class_names)):
        idx = (y_true == i)
        if idx.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((y_pred[idx] == i).mean())

    plt.figure(figsize=(10, 4))
    plt.bar(class_names, accs)
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)

# confusion_matrix.py와 동일한 방식의 로더
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
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Results file for model '{model_name}' not found. Tried: {', '.join(candidates)}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # NEW
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["baseline", "resnet18", "efficientnet"],
        help="Which model results to use for class-wise accuracy.",
    )
    args = parser.parse_args()

    # with open("results_resnet18.json", "r") as f:
    #     results = json.load(f)

    # plot_class_accuracy(
    #     results["y_true"],
    #     results["y_pred"],
    #     results["class_names"],
    #     "plots/resnet18_class_accuracy.png",
    # )

    # CHANGED: 선택된 모델 결과 읽기
    results = load_results_for_model(args.model)

    save_path = os.path.join("plots", f"{args.model}_class_accuracy.png")  # NEW
    plot_class_accuracy(
        results["y_true"],
        results["y_pred"],
        results["class_names"],
        save_path,
    )
    print(f"✅ Saved class-wise accuracy to {save_path}")

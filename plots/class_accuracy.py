import json
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    with open("results_resnet18.json", "r") as f:
        results = json.load(f)

    plot_class_accuracy(
        results["y_true"],
        results["y_pred"],
        results["class_names"],
        "plots/resnet18_class_accuracy.png",
    )

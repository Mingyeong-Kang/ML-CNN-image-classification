import json
import os
import matplotlib.pyplot as plt

## Function to load training logs
def load_history(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

## Function to plot accuracy and loss
def plot_curves(history, model_name, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    ## Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/{model_name}_accuracy.png")
    plt.close()

    ## Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_dir}/{model_name}_loss.png")
    plt.close()

    print(f"✅ Saved plots for {model_name} in {save_dir}/")

## Main
def main():
    ## List of models to plot
    models = ["baseline", "resnet18"]  

    for model_name in models:
        file_path = f"results/results_{model_name}.json"
        if os.path.exists(file_path):
            history = load_history(file_path)
            plot_curves(history, model_name)
        else:
            print(f"⚠️ Results file not found: {file_path}")

if __name__ == "__main__":
    main()

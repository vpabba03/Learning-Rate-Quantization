import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse


def plot_training(model, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(f"../trained_model_weights/resnet{model}_metrics.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metrics file for ResNet-{model} not found.")
        return
    except json.JSONDecodeError:
        print("Error: Metrics file is not a valid JSON.")
        return

    learning_rates = []
    losses = []
    accuracies = []
    epochs = []

    for lr, metrics in data.items():
        for epoch, (loss, accuracy) in enumerate(zip(metrics['loss'], metrics['accuracy']), 1):
            learning_rates.append(float(lr))
            losses.append(loss)
            accuracies.append(accuracy)
            epochs.append(epoch)

    df = pd.DataFrame({
        'Learning Rate': learning_rates,
        'Epoch': epochs,
        'Loss': losses,
        'Accuracy (%)': accuracies
    })

    plt.figure(figsize=(15, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    for lr in sorted(df['Learning Rate'].unique()):
        subset = df[df['Learning Rate'] == lr]
        plt.plot(subset['Epoch'], subset['Loss'], label=f'LR = {lr}', marker='o')
    plt.title(f'ResNet-{model} Training Loss by Learning Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(title='Learning Rates')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for lr in sorted(df['Learning Rate'].unique()):
        subset = df[df['Learning Rate'] == lr]
        plt.plot(subset['Epoch'], subset['Accuracy (%)'], label=f'LR = {lr}', marker='o')
    plt.title(f'ResNet-{model} Training Accuracy by Learning Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(title='Learning Rates')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"resnet{model}_training_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics for ResNet models.")
    parser.add_argument("--model", type=int, help="ResNet model number (e.g., 18, 50).")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../imgs/training_visualizations", 
        help="Directory to save the output plot. Default is '../../imgs/training_visualizations'."
    )
    args = parser.parse_args()

    plot_training(args.model, args.output_dir)
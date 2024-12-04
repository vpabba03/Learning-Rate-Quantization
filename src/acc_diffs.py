import os
import argparse
import pandas as pd
import re
import matplotlib.pyplot as plt

def extract_lr(file):
    """Extract the learning rate from the weights file name."""
    match = re.search(r'lr_(0.[\d]+)', file)
    return float(match.group(1)) if match else None

def preprocess_accuracy_bits(experiments_csv):
    """Preprocess the accuracy bits data from the experiments CSV."""
    experiments = pd.read_csv(experiments_csv)

    # Extract relevant columns
    accuracy_bits = experiments[
        ['Weights File', 'Model Name', 'Original Top1 Accuracy', 'Quantized Top1 Accuracy', 
         'Original Top5 Accuracy', 'Quantized Top5 Accuracy', 'Bits']
    ]

    # Add Learning Rate column
    accuracy_bits.loc[:, 'Learning Rate'] = accuracy_bits['Weights File'].apply(extract_lr)

    # Calculate accuracy differences
    accuracy_bits.loc[:, 'Top1 Accuracy Diff'] = accuracy_bits['Original Top1 Accuracy'] - accuracy_bits['Quantized Top1 Accuracy']
    accuracy_bits.loc[:, 'Top5 Accuracy Diff'] = accuracy_bits['Original Top5 Accuracy'] - accuracy_bits['Quantized Top5 Accuracy']
    
    return accuracy_bits

def plot_diffs(model, accuracy_bits, output_dir):
    """Generate and save plots for accuracy differences."""
    # Filter data for the specified model
    accuracy_bits_resnet = accuracy_bits[accuracy_bits['Model Name'] == f'resnet{model}']

    # Create pivot tables for Top1 and Top5 accuracy differences
    df_pivot_1 = accuracy_bits_resnet.pivot(index='Bits', columns='Learning Rate', values='Top1 Accuracy Diff')
    df_pivot_5 = accuracy_bits_resnet.pivot(index='Bits', columns='Learning Rate', values='Top5 Accuracy Diff')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    filepath_1 = os.path.join(output_dir, f"resnet{model}_top1_accuracy_diff.png")
    filepath_5 = os.path.join(output_dir, f"resnet{model}_top5_accuracy_diff.png")

    # Plot Top1 Accuracy Diff
    df_pivot_1.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title(f'ResNet-{model} Top1 Accuracy Diff by Bits and Learning Rate', fontsize=16)
    plt.xlabel('Bits', fontsize=12)
    plt.ylabel('Top1 Accuracy Diff', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.legend(title='Learning Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filepath_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top1 Accuracy Diff plot saved at: {filepath_1}")

    # Plot Top5 Accuracy Diff
    df_pivot_5.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title(f'ResNet-{model} Top5 Accuracy Diff by Bits and Learning Rate', fontsize=16)
    plt.xlabel('Bits', fontsize=12)
    plt.ylabel('Top5 Accuracy Diff', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.legend(title='Learning Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filepath_5, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top5 Accuracy Diff plot saved at: {filepath_5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot accuracy differences for a ResNet model.")
    parser.add_argument(
        "--model", 
        type=int, 
        required=True, 
        help="The ResNet model number (e.g., 18, 50)."
    )
    parser.add_argument(
        "--experiments-csv", 
        type=str, 
        required=True, 
        help="Path to the experiments CSV file (e.g., Q1_Project_Experiments.csv)."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default='../imgs/accuracy_differences', 
        help="Directory to save the output plots."
    )

    args = parser.parse_args()

    # Preprocess accuracy bits data
    accuracy_bits = preprocess_accuracy_bits(args.experiments_csv)

    # Generate and save plots
    plot_diffs(args.model, accuracy_bits, args.output_dir)
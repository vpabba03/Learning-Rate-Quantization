import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def extract_weights_summary(file_path):
    model = torch.load(file_path, map_location=torch.device('cpu'))

    # Extract state dictionary
    state_dict = model.state_dict() if hasattr(model, 'state_dict') else model

    # Parse file path details
    model_match = re.search(r'wfresnet([^_]+)', file_path)
    model_name = model_match.group(1) if model_match else "Unknown Model"

    lr_match = re.search(r'lr_([0-9.]+)(?=\.pth)', file_path)
    learning_rate = lr_match.group(1) if lr_match else "Unknown LR"

    bit_match = re.search(r'_b(\d+)', file_path)
    bits = bit_match.group(1) if bit_match else "Unknown Bits"

    def extract_resnet_layers(state_dict):
        layer_weights = {}

        conv1_weights = [w for k, w in state_dict.items() if 'conv1.weight' in k]
        if conv1_weights:
            layer_weights['conv1.weight'] = conv1_weights[0]

        for layer_idx in range(1, 5):
            layer_key = f'layer{layer_idx}'

            layer_blocks = [
                k for k in state_dict.keys()
                if k.startswith(f'{layer_key}.') and 'conv' in k
            ]

            block_conv_weights = {}
            for block_key in layer_blocks:
                match = re.search(rf'{layer_key}\.(\d+)\.conv(\d+)\.weight', block_key)
                if match:
                    block_num = match.group(1)
                    conv_num = match.group(2)
                    key = f'{layer_key}.{block_num}.conv{conv_num}.weight'
                    block_conv_weights[key] = state_dict[block_key]

            for key in sorted(block_conv_weights.keys()):
                layer_weights[key] = block_conv_weights[key]

        return layer_weights

    weight_layers = extract_resnet_layers(state_dict)

    weight_summaries = {}
    for key, value in weight_layers.items():
        weights = value.cpu().numpy()
        weight_summaries[key] = {"std": np.std(weights)}

    return {
        "model_name": f'{model_name}',
        "learning_rate": learning_rate,
        "bits": bits,
        "weight_summaries": weight_summaries
    }


def visualize_weight_summary_comparison(weights_summary_quant, weights_summary_original, output_dir):
    layer_names = list(weights_summary_quant['weight_summaries'].keys())
    stds_quant = [stats["std"] for stats in weights_summary_quant['weight_summaries'].values()]
    stds_original = [stats["std"] for stats in weights_summary_original['weight_summaries'].values()]

    bits_suffix = f"{weights_summary_quant.get('bits', 'NA')}" if 'bits' in weights_summary_quant else ""
    filename = f"resnet{weights_summary_quant['model_name']}_lr{weights_summary_quant['learning_rate']}_b{bits_suffix}_comparison.png"
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(14, 7))

    plt.plot(np.arange(len(layer_names)), stds_quant, label=f"Quantized (Bits: {weights_summary_quant['bits']})",
             marker="o", color='blue', linestyle='-', linewidth=2)
    plt.plot(np.arange(len(layer_names)), stds_original, label="Original (Full Precision)",
             marker="s", color='red', linestyle='--', linewidth=2)

    plt.xticks(np.arange(len(layer_names)), np.arange(1, len(layer_names) + 1), rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Standard Deviation")
    plt.title((f"Weight Distribution Comparison - ResNet-{weights_summary_quant['model_name']} "
               f"(Learning Rate: {weights_summary_quant['learning_rate']})"),
              fontsize=14)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {filepath}")
    return filepath


def extract_model_details(filename):
    model_match = re.search(r'(wfresnet\d+)', filename)
    model_name = model_match.group(1) if model_match else None

    lr_match = re.search(r'lr_(\d+\.\d+)', filename)
    learning_rate = float(lr_match.group(1)) if lr_match else None

    bits_match = re.search(r'_b(\d+)_', filename)
    bits = int(bits_match.group(1)) if bits_match else None

    return {
        'model_name': model_name,
        'learning_rate': learning_rate,
        'bits': bits
    } if model_name and learning_rate is not None else None


def main(quantized_dir, original_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    original_models_map = {}
    for original_file in os.listdir(original_dir):
        if original_file.endswith('.pth'):
            lr_match = re.search(r'lr_(\d+\.\d+)', original_file)
            model_match = re.search(r'(resnet\d+)', original_file)

            if lr_match and model_match:
                model_name = model_match.group(1)
                lr = float(lr_match.group(1))

                if model_name not in original_models_map:
                    original_models_map[model_name] = {}
                if lr not in original_models_map[model_name]:
                    original_models_map[model_name][lr] = os.path.join(original_dir, original_file)

    for pt_file in os.listdir(quantized_dir):
        if pt_file.endswith('.pt'):
            file_path = os.path.join(quantized_dir, pt_file)

            try:
                details = extract_model_details(pt_file)

                model_name = details['model_name'][2:] if details and details['model_name'].startswith('wf') else details['model_name']

                if (details and
                        model_name in original_models_map and
                        details['learning_rate'] in original_models_map[model_name]):

                    original_model_path = original_models_map[model_name][details['learning_rate']]

                    weights_quant = extract_weights_summary(file_path)
                    weights_original = extract_weights_summary(original_model_path)

                    weights_quant['bits'] = details['bits']

                    visualize_weight_summary_comparison(
                        weights_quant,
                        weights_original,
                        output_dir
                    )
                else:
                    print(f"No matching original model found for {pt_file}")

            except Exception as e:
                print(f"Error processing {pt_file}: {e}")

    print(f"Plots have been saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare weights of quantized and original ResNet models.")
    parser.add_argument("--quantized_dir", required=True, help="Directory containing quantized models.")
    parser.add_argument("--original_dir", required=True, help="Directory containing original models.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output plots.")
    args = parser.parse_args()

    main(args.quantized_dir, args.original_dir, args.output_dir)

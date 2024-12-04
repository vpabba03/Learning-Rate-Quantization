import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import os
import sys
import argparse
import json

# Add parent directory for custom imports
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from data_loaders import data_loader

def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10.")
    parser.add_argument(
        '--model', '-m',
        type=str, 
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], 
        required=True,
        help="ResNet model to use (resnet18, resnet34, resnet50, resnet101)"
    )
    parser.add_argument(
        '--learning_rates', '-lr',
        type=float, 
        nargs='+', 
        default=[0.001, 0.0001, 0.01],
        help="Learning rates to experiment with (default: [0.001, 0.0001, 0.01])"
    )
    parser.add_argument(
        '--batch_size', '-bs',
        type=int, 
        default=128,
        help="Batch size for training (default: 128)"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../trained_model_weights',
        help="Directory to save trained models and metrics (default: '../trained_model_weights')"
    )
    args = parser.parse_args()

    # Hyperparameters
    learning_rates = args.learning_rates
    batch_size = args.batch_size
    num_epochs = args.epochs
    output_dir = args.output_dir

    # Prepare data loaders
    train_loader, test_loader = data_loader('CIFAR10', batch_size, num_workers=8)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    metrics = {lr: {'loss': [], 'accuracy': []} for lr in learning_rates}

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")

        # Initialize model
        if args.model == 'resnet18':
            model = torchvision.models.resnet18(weights=None)
        elif args.model == 'resnet34':
            model = torchvision.models.resnet34(weights=None)
        elif args.model == 'resnet50':
            model = torchvision.models.resnet50(weights=None)
        elif args.model == 'resnet101':
            model = torchvision.models.resnet101(weights=None)

        model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
        model = model.to(device)

        # Define optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total

            metrics[lr]['loss'].append(epoch_loss)
            metrics[lr]['accuracy'].append(epoch_accuracy)

            print(f"LR {lr} | Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

        # Save weights
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, f"{args.model}_lr_{lr}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model with LR {lr} saved to {model_save_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{args.model}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()

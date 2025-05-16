import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def main():
    # Fix random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Detect the device for computation (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ========== Data preprocessing and loading ==========
    # CIFAR-100 dataset mean and standard deviation values for normalization
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    # Data augmentation and transformation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Randomly crop 32x32 images with padding
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.RandAugment(num_ops=2, magnitude=10), # Apply random augmentations
        transforms.ToTensor(), # Convert images to tensor
        transforms.Normalize(mean, std) # Normalize using mean and std
    ])

    # Data transformation for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load CIFAR-100 training and testing datasets
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Create DataLoaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4,
                                              pin_memory=True)

    # ========== Model definition (using pre-trained ResNet34 weights) ==========
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) # Load ResNet34 pre-trained on ImageNet
    model.fc = nn.Linear(model.fc.in_features, 100) # Replace the fully connected layer for CIFAR-100 classification
    model.to(device) # Move the model to the appropriate device

    # ========== Loss function and optimizer ==========
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # Cross-entropy loss with label smoothing
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4) # Stochastic Gradient Descent

    num_epochs = 100 # Number of training epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) # Cosine annealing learning rate

    # ========== Mixup and CutMix augmentation functions ==========
    def mixup_data(x, y, alpha=1.0):
        # Mixup data augmentation
        if alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha) # Sample mixup ratio
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device) # Shuffle indices
        mixed_x = lam * x + (1 - lam) * x[index, :] # Combine images
        y_a, y_b = y, y[index] # Original and shuffled targets
        return mixed_x, y_a, y_b, lam

    def cutmix_data(x, y, alpha=1.0):
        # CutMix data augmentation
        if alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha) # Sample CutMix ratio
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device) # Shuffle indices

        # Define CutMix region
        height, width = x.size(2), x.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_rat)
        cut_w = int(width * cut_rat)

        # Randomly select CutMix region
        cy = np.random.randint(height)
        cx = np.random.randint(width)

        y1 = np.clip(cy - cut_h // 2, 0, height)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        x2 = np.clip(cx + cut_w // 2, 0, width)

        # Apply CutMix
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (height * width)) # Adjust lambda based on CutMix region
        y_a, y_b = y, y[index] # Original and shuffled targets
        return x, y_a, y_b, lam

    def mixup_cutmix_criterion(pred, y_a, y_b, lam):
        # Compute loss for Mixup and CutMix
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # ========== Training and testing functions ==========
    def train_one_epoch(epoch):
        model.train() # Set model to training mode
        total_loss = 0.0
        normal_correct = 0
        normal_total = 0

        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() # Zero gradients

            rand_val = np.random.rand()
            if rand_val < 0.33:
                # # Apply Mixup
                mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=1.0)
                outputs = model(mixed_x)
                loss = mixup_cutmix_criterion(outputs, y_a, y_b, lam)
            elif rand_val < 0.66:
                # # Apply CutMix
                cm_x, y_a, y_b, lam = cutmix_data(inputs, targets, alpha=1.0)
                outputs = model(cm_x)
                loss = mixup_cutmix_criterion(outputs, y_a, y_b, lam)
            else:
                # No augmentation (normal training)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1) # Get predictions
                normal_correct += predicted.eq(targets).sum().item() # Count correct predictions
                normal_total += targets.size(0)

            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_dataset) # Average loss

        # Only the accuracy of the normal batch statistics is reported.
        if normal_total > 0:
            acc = 100. * normal_correct / normal_total
        else:
            acc = 0.0  # If there are no normal batches in the current round, set to 0 or do not display

        print(f"Epoch: {epoch} | Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}% (Normal batch only)")
        return avg_loss, acc

    def test_one_epoch(epoch):
        model.eval() # Set model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # No gradient computation
            for (inputs, targets) in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1) # Get predictions
                correct += predicted.eq(targets).sum().item() # Count correct predictions
                total += targets.size(0)

        acc = 100. * correct / total # Accuracy
        avg_loss = total_loss / total # Average loss
        print(f"Epoch: {epoch} | Test Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")
        return avg_loss, acc

    # ========== Training loop and model saving ==========
    best_acc = 0.0 # Track best accuracy
    best_model_path = "best_resnet34_cifar100.pth" # Path to save the best model

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        test_loss, test_acc = test_one_epoch(epoch)
        scheduler.step() # Update learning rate scheduler

        # Save the best model based on accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Test Acc: {best_acc:.2f}%")

    # ========== Visualization of predictions ==========
    # Load model parameters
    model.load_state_dict(torch.load(best_model_path)) # Load best model
    model.eval()

    def imshow(img):
        # Convert tensor to numpy and denormalize
        img = img.cpu()
        img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = img.clamp(0, 1)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # Transpose for correct image display
        plt.axis('off')

    # Display predictions on test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    images_to_show = images[:8]
    labels_to_show = labels[:8]

    with torch.no_grad():
        outputs = model(images_to_show)
        _, preds = outputs.max(1)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        imshow(images_to_show[i])
        plt.title(f"Pred: {preds[i].item()}, True: {labels_to_show[i].item()}")
    plt.tight_layout()
    plt.show()

    print("Done! Best Test Accuracy: {:.2f}%".format(best_acc))


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model import VisionTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = VisionTransformer(
        n_channels=1,
        embed_dim=64,
        n_layers=6,
        n_attention_heads=4,
        forward_mul=2,
        image_size=28,
        patch_size=7,
        n_classes=10,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# of parameters: {total_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        train_accuracies.append(train_acc)
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        test_acc = 100 * correct_test / total_test
        test_accuracies.append(test_acc)
        
        print(f"epoch {epoch+1}/{epochs} - train: {train_acc}% - test: {test_acc}%")

    print(f"final test: {test_accuracies[-1]}%")

    model_path = 'vit_mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"model file size: {file_size_mb}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', marker='s')
    plt.title('ViT on MNIST: Training and Test Accuracy over 20 Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig('vit_accuracy_curves.png')
    plt.close()

if __name__ == '__main__':
    main()
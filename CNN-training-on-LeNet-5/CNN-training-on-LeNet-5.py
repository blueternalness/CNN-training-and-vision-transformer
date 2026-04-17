import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import os

Q_B = True  
Q_C = True  
Q_D = True  

CLASS_NAMES = {
    'MNIST': [str(i) for i in range(10)],
    'Fashion-MNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'CIFAR-10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}

class LeNet5(nn.Module):
    def __init__(self, in_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def run_experiment(dataset_name, in_channels, train_loader, test_loader, epochs, lr, optimizer_type='SGD', weight_decay=0, runs=5, save_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{runs} times, {optimizer_type}, LR: {lr}, WD: {weight_decay}")
    
    all_runs_test_acc = []
    best_test_acc_overall = 0
    best_run_train_curve, best_run_test_curve = [], []

    for run in range(runs):
        model = LeNet5(in_channels=in_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
        train_curve, test_curve = [], []
        best_acc_this_run = 0
        
        for epoch in range(epochs):
            train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            train_curve.append(train_acc)
            test_curve.append(test_acc)
            
            if test_acc > best_acc_this_run:
                best_acc_this_run = test_acc
                if test_acc > best_test_acc_overall and save_name is not None:
                    torch.save(model.state_dict(), f"{save_name}.pth")
                
        all_runs_test_acc.append(best_acc_this_run)
        print(f"{run+1}/{runs} - best: {best_acc_this_run:.2f}%")
        
        if best_acc_this_run > best_test_acc_overall:
            best_test_acc_overall = best_acc_this_run
            best_run_train_curve = train_curve
            best_run_test_curve = test_curve

    print(f"{dataset_name} Results: best: {best_test_acc_overall:.2f}%, mean: {np.mean(all_runs_test_acc):.2f}%")
    return best_test_acc_overall, best_run_train_curve, best_run_test_curve

def plot_curves(train_curve, test_curve, title):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_curve)+1), train_curve, label='Train Accuracy')
    plt.plot(range(1, len(test_curve)+1), test_curve, label='Test Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    safe_title = title.replace(' ', '_').replace('/', '_')
    plt.savefig(f"{safe_title}_accuracy.png", bbox_inches='tight')
    plt.close()

def get_predictions_and_probs(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images_device = images.to(device)
            outputs = model(images_device)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_images)

def analyze_confusion_matrix(y_true, y_pred, images, dataset_name):
    classes = CLASS_NAMES[dataset_name]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Normalized Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{dataset_name}_confusion_matrix.png", bbox_inches='tight')
    plt.close()
    
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, -1)
    flat_indices = np.argsort(cm_off_diag, axis=None)[-3:][::-1]
    top_pairs = []
    for idx in flat_indices:
        true_idx, pred_idx = np.unravel_index(idx, cm_off_diag.shape)
        top_pairs.append((true_idx, pred_idx, cm[true_idx, pred_idx]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Top 3 Confused Pairs - {dataset_name}', fontsize=16)
    
    for i, (t_idx, p_idx, _) in enumerate(top_pairs):
        misclassified_indices = np.where((y_true == t_idx) & (y_pred == p_idx))[0]
        if len(misclassified_indices) == 0:
            axes[i].axis('off')
            continue
        img = images[misclassified_indices[0]]
        img = img / 2 + 0.5 
        if dataset_name == 'CIFAR-10':
            img = np.transpose(img, (1, 2, 0))
        else:
            img = img.squeeze()
            
        img = np.clip(img, 0, 1)
        axes[i].imshow(img, cmap='gray' if dataset_name != 'CIFAR-10' else None)
        axes[i].set_title(f"True: {classes[t_idx]}\nPred: {classes[p_idx]}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_top3_confusions.png", bbox_inches='tight')
    plt.close()

def analyze_roc_auc_cifar(y_true, y_probs):
    classes = CLASS_NAMES['CIFAR-10']
    
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {auc(fpr, tpr):.3f})')

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest Multiclass ROC - CIFAR-10')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("CIFAR10_ROC_Curve.png", bbox_inches='tight')
    plt.close()

    weighted_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    print(f"Weighted AUC: {weighted_auc:.4f}")

def apply_symmetric_label_noise(targets, epsilon, num_classes=10):
    if epsilon == 0.0: return targets.copy()
    noisy_targets = targets.copy()
    flip_mask = np.random.rand(len(targets)) < epsilon
    
    for i in range(len(targets)):
        if flip_mask[i]:
            available_classes = [c for c in range(num_classes) if c != targets[i]]
            noisy_targets[i] = np.random.choice(available_classes)
    return noisy_targets

def plot_noise_confusion_matrix(clean_labels, noisy_labels, epsilon, title):
    cm = confusion_matrix(clean_labels, noisy_labels, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Reds')
    plt.title(f'{title} (epsilon = {int(epsilon*100)}%)')
    plt.ylabel('True Clean Label')
    plt.xlabel('Assigned Noisy Label')
    plt.savefig(f"Noise_Matrix_eps_{int(epsilon*100)}.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_gray = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_color = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64
    b_epochs = 15

    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_gray)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_gray)
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    fmnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform_gray)
    fmnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform_gray)
    fmnist_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=False)

    cifar_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_color)
    cifar_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_color)
    cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    if Q_B: # Question B
        print("Problem B")
        settings = [
            {'opt': 'SGD', 'lr': 0.01, 'wd': 0},
            {'opt': 'Adam', 'lr': 0.001, 'wd': 0},
            {'opt': 'Adam', 'lr': 0.001, 'wd': 1e-4}
        ]
        for i, p in enumerate(settings):
            _, tr, te = run_experiment("MNIST Setting", 1, mnist_train_loader, mnist_test_loader, b_epochs, p['lr'], p['opt'], p['wd'], runs=5) 
            plot_curves(tr, te, f"MNIST Setting {i+1}")

        _, tr, te = run_experiment("MNIST (Opt)", 1, mnist_train_loader, mnist_test_loader, b_epochs, 0.001, 'Adam', 0, 5, "best_mnist")
        plot_curves(tr, te, "MNIST Optimal")
        
        _, tr, te = run_experiment("Fashion-MNIST (Opt)", 1, fmnist_train_loader, fmnist_test_loader, b_epochs, 0.001, 'Adam', 0, 5, "best_fmnist")
        plot_curves(tr, te, "Fashion-MNIST Optimal")
        
        _, tr, te = run_experiment("CIFAR-10 (Opt)", 3, cifar_train_loader, cifar_test_loader, b_epochs+5, 0.001, 'Adam', 1e-4, 5, "best_cifar")
        plot_curves(tr, te, "CIFAR-10 Optimal")

    if Q_C: # Question C
        print("Problem C")
        m_mnist = LeNet5(1).to(device)
        m_fmnist = LeNet5(1).to(device)
        m_cifar = LeNet5(3).to(device)
        
        if os.path.exists("best_mnist.pth"): m_mnist.load_state_dict(torch.load("best_mnist.pth"))
        if os.path.exists("best_fmnist.pth"): m_fmnist.load_state_dict(torch.load("best_fmnist.pth"))
        if os.path.exists("best_cifar.pth"): m_cifar.load_state_dict(torch.load("best_cifar.pth"))

        y_true, y_pred, _, imgs = get_predictions_and_probs(m_mnist, mnist_test_loader, device)
        analyze_confusion_matrix(y_true, y_pred, imgs, 'MNIST')

        y_true, y_pred, _, imgs = get_predictions_and_probs(m_fmnist, fmnist_test_loader, device)
        analyze_confusion_matrix(y_true, y_pred, imgs, 'Fashion-MNIST')

        y_true, y_pred, y_probs, imgs = get_predictions_and_probs(m_cifar, cifar_test_loader, device)
        analyze_confusion_matrix(y_true, y_pred, imgs, 'CIFAR-10')
        analyze_roc_auc_cifar(y_true, y_probs)

    if Q_D: # Question D
        print("Problem D")
        
        clean_train_targets = np.array(fmnist_train.targets)
        
        noisy_targets_50 = apply_symmetric_label_noise(clean_train_targets, 0.5)
        plot_noise_confusion_matrix(clean_train_targets, noisy_targets_50, 0.5, "SLN Transition Matrix")
        epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]
        d_runs, d_epochs = 5, 10
        mean_accs, std_accs = [], []

        for eps in epsilons:
            print(f"noise level: {eps*100}%")
            run_accs = []
            
            for r in range(d_runs):
                noisy_targets = apply_symmetric_label_noise(clean_train_targets, eps)
                noisy_train_dataset = copy.deepcopy(fmnist_train)
                noisy_train_dataset.targets = torch.tensor(noisy_targets)
                noisy_train_loader = DataLoader(noisy_train_dataset, batch_size=64, shuffle=True)
                model = LeNet5(1).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                best_acc = 0
                for epoch in range(d_epochs):
                    train_epoch(model, noisy_train_loader, nn.CrossEntropyLoss(), optimizer, device)
                    acc = evaluate(model, fmnist_test_loader, device)
                    if acc > best_acc: best_acc = acc
                        
                run_accs.append(best_acc)
                print(f"  Run {r+1}/{d_runs} - Acc: {best_acc:.2f}%")
                
            mean_accs.append(np.mean(run_accs))
            std_accs.append(np.std(run_accs))

        plt.figure(figsize=(8, 5))
        plt.errorbar(np.array(epsilons) * 100, mean_accs, yerr=std_accs, marker='o', capsize=5, color='b', ecolor='r')
        plt.title('Testing Accuracy vs. Label Noise Level (Fashion-MNIST)')
        plt.xlabel('Noise Level $\epsilon$ (%)')
        plt.ylabel('Mean Testing Accuracy (%)')
        plt.xticks(np.array(epsilons) * 100)
        plt.grid(True)
        plt.savefig("Noise_Accuracy_Degradation_Curve.png", bbox_inches='tight')
        plt.close()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import random

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task 1: Setup and Model Preparation

# 1. Load MNIST test dataset normalized to [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),  # converts to [0,1] tensor
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# Training function
def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Load training data for training the model
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model (uncomment if model not pretrained)
# train(model, train_loader, optimizer, criterion, epochs=5)

# Alternatively, try to load pretrained weights if available:
# try:
#     model.load_state_dict(torch.load('mnist_cnn.pth'))
#     print("Model loaded")
# except:
#     train(model, train_loader, optimizer, criterion, epochs=5)
#     torch.save(model.state_dict(), 'mnist_cnn.pth')

# For now, let's train quickly:
train(model, train_loader, optimizer, criterion, epochs=3)

# Evaluate model accuracy on test set
def test_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total

test_acc = test_accuracy(model, test_loader)
print(f"Test accuracy: {test_acc*100:.2f}%")

assert test_acc > 0.98, "Model accuracy should be > 98%"

# 3. Select 100 random test images correctly classified initially
model.eval()
all_test_images = []
all_test_labels = []
all_test_preds = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)
        for i in range(data.size(0)):
            if preds[i] == target[i]:
                all_test_images.append(data[i].cpu())
                all_test_labels.append(target[i].cpu())
                all_test_preds.append(preds[i].cpu())
            if len(all_test_images) >= 100:
                break
        if len(all_test_images) >= 100:
            break

all_test_images = torch.stack(all_test_images)  # (100,1,28,28)
all_test_labels = torch.tensor(all_test_labels)
all_test_preds = torch.tensor(all_test_preds)

print(f"Selected {len(all_test_images)} correctly classified test images.")

# 4. Define function to compute predictions and loss
def predict_and_loss(model, images, labels):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    preds = outputs.argmax(dim=1)
    return preds.cpu(), loss.item()

# Task 2: Implement Targeted FGSM Attack

def targeted_fgsm_attack(image, epsilon, target_label, model):
    """
    image: torch.Tensor of shape (1, 1, 28, 28)
    epsilon: scalar
    target_label: int scalar (target class)
    model: torch.nn.Module
    """
    image = image.to(device)
    image.requires_grad = True
    target = torch.tensor([target_label]).to(device)

    output = model(image)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()

    # Compute perturbation: negative sign of gradient for targeted attack
    perturbation = -epsilon * image.grad.sign()
    adv_image = image + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)  # Keep image in [0,1]
    return adv_image.detach()

# For each of the 100 images, randomly select a target label y_t != y (true label)
random.seed(42)
target_labels = []
for true_label in all_test_labels:
    choices = list(range(10))
    choices.remove(true_label.item())
    target_labels.append(random.choice(choices))
target_labels = torch.tensor(target_labels)

# Choose epsilon values as typical for FGSM, e.g., [0, 0.05, 0.1, 0.15, 0.2]
epsilons = [0, 0.05, 0.1, 0.15, 0.2]

success_rates = []
avg_linf_norms = []

for eps in epsilons:
    success_count = 0
    linf_norms = []
    adv_images = []

    for i in range(len(all_test_images)):
        img = all_test_images[i].unsqueeze(0)  # (1,1,28,28)
        true_label = all_test_labels[i].item()
        target_label = target_labels[i].item()

        adv_img = targeted_fgsm_attack(img, eps, target_label, model)

        # Predict on adversarial example
        pred, _ = predict_and_loss(model, adv_img, torch.tensor([true_label]))
        pred_label = model(adv_img.to(device)).argmax(dim=1).item()

        if pred_label == target_label:
            success_count += 1

        linf_norm = (adv_img - img).abs().max().item()
        linf_norms.append(linf_norm)

        if eps == epsilons[-1] and i < 5:  # Save 5 examples for visualization at max epsilon
            adv_images.append((img.squeeze().cpu(), adv_img.squeeze().cpu(), true_label, target_label, pred_label))

    success_rate = success_count / len(all_test_images)
    avg_linf = np.mean(linf_norms)
    success_rates.append(success_rate)
    avg_linf_norms.append(avg_linf)

    print(f"Epsilon: {eps:.3f} | Success Rate: {success_rate*100:.2f}% | Avg L-inf norm: {avg_linf:.4f}")

# Visualization of 5 examples at max epsilon
# Visualization of 5 examples at max epsilon
fig, axes = plt.subplots(5, 3, figsize=(10, 15))
fig.suptitle(f"Targeted FGSM Attack Examples at epsilon={epsilons[-1]}")

for i, (orig, adv, true_label, target_label, pred_label) in enumerate(adv_images):
    # Original image
    axes[i, 0].imshow(orig.detach().numpy(), cmap='gray')
    axes[i, 0].set_title(f"Original\nTrue: {true_label}")
    axes[i, 0].axis('off')

    # Adversarial image
    axes[i, 1].imshow(adv.detach().numpy(), cmap='gray')
    axes[i, 1].set_title(f"Adversarial\nTarget: {target_label}\nPred: {pred_label}")
    axes[i, 1].axis('off')

    # Perturbation
    diff = (adv - orig).abs()
    axes[i, 2].imshow(diff.detach().numpy(), cmap='gray')
    axes[i, 2].set_title("Perturbation")
    axes[i, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


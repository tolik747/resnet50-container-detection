import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Підготовка трансформацій (без аугментації!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. Завантаження даних
train_dataset = datasets.ImageFolder('./dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('./dataset/val', transform=transform)
test_dataset = datasets.ImageFolder('./dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Кількість класів: {len(train_dataset.classes)} -> {train_dataset.classes}")

#3. Завантаження моделі ResNet-50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)

# Заморожуємо всі шари
for param in model.parameters():
    param.requires_grad = False

# Заміна останнього шару на 11 класів
model.fc = nn.Linear(model.fc.in_features, 11)

model = model.to(device)

# 4. Налаштування оптимізатора та функції втрат
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 5. Тренування моделі
num_epochs = 10
best_val_accuracy = 0.0

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    epoch_val_acc = correct / len(val_loader.dataset)
    val_accuracies.append(epoch_val_acc)

    print(f"Епоха [{epoch+1}/{num_epochs}], Тренувальний Loss: {epoch_loss:.4f}, Валідаційний Loss: {epoch_val_loss:.4f}, Валідаційна Точність: {epoch_val_acc:.4f}")

    # save best model resnet
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        torch.save(model.state_dict(), 'best_resnet50_model.pth')
        print("save a best model resnet50")

# 6. load best model
model.load_state_dict(torch.load('best_resnet50_model.pth'))

# 7. testing best model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 8. analiz
print("klasification")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

print("confusion matrix")
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)

# 9. graff loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='trein Loss')
plt.plot(val_losses, label='valid Loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Графік втрат')
plt.show()

# 10. Побудова графіку Валідаційної Точності
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Валідаційна Точність')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Точність')
plt.title('Графік валідаційної точності')
plt.show()

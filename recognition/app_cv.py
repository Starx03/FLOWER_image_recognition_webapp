import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Use 'test' set (6149 images) for training and 'train' set (1020 images) for validation
train_dataset = datasets.Flowers102(
    root="./data", split="test", download=True, transform=data_transforms["train"]
)
val_dataset = datasets.Flowers102(
    root="./data", split="train", download=True, transform=data_transforms["val"]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = 102

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
  param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=1e-3)


def train_one_epoch(model, dataloader, optimizer, criterion):
  model.train()
  running_loss, running_corrects = 0.0, 0
  for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)

    loss.backward()
    optimizer.step()

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

  epoch_loss = running_loss / len(dataloader.dataset)
  epoch_acc = running_corrects.double() / len(dataloader.dataset)
  return epoch_loss, epoch_acc


print("\n--- STAGE 1: Training Head (5 Epochs) ---")
for epoch in range(5):
  loss, acc = train_one_epoch(model, train_loader, optimizer_stage1, criterion)
  print(f"Epoch {epoch+1}/5 - Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

print("\n--- STAGE 2: Fine-Tuning Layer 4 (8 Epochs) ---")
for param in model.layer4.parameters():
  param.requires_grad = True

optimizer_stage2 = optim.AdamW(
    [
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-2,
)

for epoch in range(8):
  loss, acc = train_one_epoch(model, train_loader, optimizer_stage2, criterion)
  print(
      f"Fine-Tune Epoch {epoch+1}/8 - Loss: {loss:.4f} | Accuracy:"
      f" {acc*100:.2f}%"
  )

torch.save(model.state_dict(), "flower_resnet50.pth")
print("\n🎉 Saved updated weights to 'flower_resnet50.pth'!")

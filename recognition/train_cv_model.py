import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import timm

# 1. SETUP DEVICE (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. IMAGE TRANSFORMATIONS & DATA AUGMENTATION
# Pretrained ImageNet models expect images normalized with specific mean & std
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. DOWNLOAD & LOAD DATASET
print("Downloading / Loading dataset...")
train_dataset = datasets.Flowers102(root='./data', split='train', download=True, transform=data_transforms['train'])
val_dataset = datasets.Flowers102(root='./data', split='val', download=True, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = 102
print(f"Dataset loaded! Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# 4. LOAD PRETRAINED RESNET50 (Stage 1: Feature Extraction)
print("Loading ResNet50 model with ImageNet weights...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all backbone layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classification layer for 102 flower classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# STAGE 1 OPTIMIZER: Only update weights of the final head
optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=1e-3)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in dataloader:
        # If labels start at 1, shift them down by 1. Keep 0-indexed labels as 0.
        if labels.min() > 0:
            labels = labels - 1
            
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

print("\n--- STAGE 1: Training Classification Head Only (3 Epochs) ---")
for epoch in range(5):
    loss, acc = train_one_epoch(model, train_loader, optimizer_stage1, criterion)
    print(f"Epoch {epoch+1}/5 - Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# 5. STAGE 2: FINE-TUNING DEEPER LAYERS
print("\n--- STAGE 2: Unfreezing Layer 4 for Fine-Tuning ---")

# First, unfreeze layer 4 parameters
for param in model.layer4.parameters():
    param.requires_grad = True

# Second, DEFINE the optimizer FIRST
optimizer_stage2 = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
], weight_decay=1e-2)

# Third, RUN the training loop using optimizer_stage2
for epoch in range(8):
    loss, acc = train_one_epoch(model, train_loader, optimizer_stage2, criterion)
    print(f"Fine-Tune Epoch {epoch+1}/8 - Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# 6. SAVE TRAINED WEIGHTS
torch.save(model.state_dict(), 'flower_resnet50.pth')
print("\n🎉 Model weights saved to 'flower_resnet50.pth'!")
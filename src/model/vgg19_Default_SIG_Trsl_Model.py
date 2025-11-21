# To apply transfer learning with the concept you mentioned
# (fine-tuning the top layers with a small learning rate and freezing the
# bottom layers)

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
smoker_path = './data/SMV_c_smoker/'
noncsmoke_path = './data/SMV_non_c_smoker/'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Load and preprocess data
data = []
labels = []
for folder, label in [(smoker_path, 1), (noncsmoke_path, 0)]:
    for image_file in tqdm(os.listdir(folder)):
        image = Image.open(os.path.join(folder, image_file))
        image = data_transform(image)
        data.append(image)
        labels.append(label)

X = torch.stack(data)
Y = torch.tensor(labels)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define DataLoader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize VGG19 model
vgg19 = models.vgg19(pretrained=True).to(device)

# Freeze all layers except the last fully connected layers
for param in vgg19.features.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for the binary classification task
vgg19.classifier[6] = nn.Linear(4096, 2).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(vgg19.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 5
train_loss_history = []

for epoch in range(num_epochs):
    vgg19.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = vgg19(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Test the model
vgg19.eval()
test_loss = 0.0
preds = []
true_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg19(images)
        test_loss += criterion(outputs, labels).item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds)
recall = recall_score(true_labels, preds)
f1 = f1_score(true_labels, preds)
tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
specificity = tn / (tn + fp)

# Print evaluation metrics for the model
print("Evaluation after training with VGG19:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1:.4f}")

# Plot train loss values
plt.plot(train_loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

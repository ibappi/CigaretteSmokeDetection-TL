import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import time
from fvcore.nn import FlopCountAnalysis

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
smoker_path = '../data/c_smoker/'
noncsmoke_path = '../data/non_c_smoker/'

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

# Initialize RegNet model
regnet = timm.create_model('regnetx_002', pretrained=True).to(device)

# Modify the classifier part for the binary classification task
num_features = regnet.head.fc.in_features
regnet.head.fc = nn.Linear(num_features, 2).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(regnet.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 50
train_loss_history = []

for epoch in range(num_epochs):
    regnet.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = regnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Test the model
regnet.eval()
test_loss = 0.0
preds = []
true_labels = []

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    start_event.record()

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = regnet(images)
        test_loss += criterion(outputs, labels).item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    end_event.record()
    torch.cuda.synchronize()

    inference_time_ms = start_event.elapsed_time(end_event) / len(test_loader.dataset) * 1000

test_loss /= len(test_loader.dataset)
test_accuracy = accuracy_score(true_labels, preds)
precision = precision_score(true_labels, preds)
recall = recall_score(true_labels, preds)
f1 = f1_score(true_labels, preds)
tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
specificity = tn / (tn + fp)

# Print evaluation metrics for the model
print("Evaluation after training with RegNet:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Inference Time: {inference_time_ms:.4f} ms/image")

# Plot train loss values
plt.plot(train_loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


# Calculate GFLOPS using fvcore
def calculate_gflops(model, input_tensor):
    flops = FlopCountAnalysis(model, input_tensor)
    gflops = flops.total() / 1e9
    return gflops


input_tensor = torch.randn(1, 3, 224, 224).to(device)
gflops = calculate_gflops(regnet, input_tensor)
print(f'GFLOPS: {gflops:.4f}')

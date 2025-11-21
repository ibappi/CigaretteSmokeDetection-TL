import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from fvcore.nn import FlopCountAnalysis


# Function to set up device for DataParallel if multiple GPUs are available
def setup_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            return torch.device('cuda:0')  # Use first GPU for DataParallel
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


csmoke = os.listdir('../data/c_smoker')
noncsmoke = os.listdir('../data/non_c_smoker')

print(csmoke[0:5])
print(noncsmoke[0:5])

# length of folder
print(len(csmoke))
print(len(noncsmoke))

# Labeling the data
label_c_smoke = [1] * len(csmoke)
label_non_csmoke = [0] * len(noncsmoke)
label = label_c_smoke + label_non_csmoke

print(len(label))
print(type(label))

# image resize and convert
smoker_path = '../data/c_smoker/'
noncsmoke_path = '../data/non_c_smoker/'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

data = []

for image_file in tqdm(csmoke):
    image = Image.open(smoker_path + image_file)
    image = data_transform(image)
    image = np.array(image)
    data.append(image)

for image_file in tqdm(noncsmoke):
    image = Image.open(noncsmoke_path + image_file)
    image = data_transform(image)
    image = np.array(image)
    data.append(image)

X = np.array(data)
Y = np.array(label)

print(X.shape, Y.shape)

# split for the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).long()

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

del train_dataset, test_dataset

# device configuration
device = setup_device()

# load model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

# Move the model to the device
model = model.to(device)

# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

print('device is', device)
print('Model is: ', model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    train_loss_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

    return train_loss_history, train_acc_history


train_loss_history, train_acc_history = train_model(model, train_loader, criterion, optimizer, num_epochs=50)
print('Loss: ')
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

print('Accuracy: ')
plt.plot(train_acc_history, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()
plt.grid(True)
plt.show()


def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()

        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        end_event.record()
        torch.cuda.synchronize()

        inference_time_ms = start_event.elapsed_time(end_event) / len(test_loader.dataset)

        test_loss = test_loss / len(test_loader)
        test_acc = correct / total
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

        # Calculate additional metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        sensitivity = recall_score(true_labels, predictions)
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        specificity = tn / (tn + fp)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Sensitivity: {sensitivity:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Inference Time: {inference_time_ms:.4f} ms/image')


evaluate_model(model, test_loader, criterion)


# Calculate the gflops
def calculate_gflops(model, input_tensor):
    flops = FlopCountAnalysis(model, input_tensor)
    gflops = flops.total() / 1e9
    return gflops


input_tensor = torch.randn(1, 3, 224, 224).to(device)
gflops = calculate_gflops(model, input_tensor)
print(f'GFLOPs: {gflops:.4f}')

print(torch.cuda.memory_summary(device=device))
print('CUDA cache clean: ', torch.cuda.empty_cache())

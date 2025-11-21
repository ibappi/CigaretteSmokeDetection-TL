import numpy as np
import os
from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from tqdm import tqdm


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


csmoke = os.listdir('./data/c_smoker')
noncsmoke = os.listdir('./data/non_c_smoker')

print(csmoke[0:5])
print(noncsmoke[0:5])

# img = mpimg.imread('./data/c_smoker/' + csmoke[1])
# plt.imshow(img)
# plt.show()

# length of folder
print(len(csmoke))
print(len(noncsmoke))

# Labeling the data
label_c_smoke = [1] * len(csmoke)
label_non_csmoke = [0] * len(noncsmoke)
labels = label_c_smoke + label_non_csmoke

print(len(labels))
print(type(labels))

# image resize and convert
smoker_path = './data/SMV_c_smoker/'
noncsmoke_path = './data/SMV_non_c_smoker/'

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
    # image = image / 255.0
    data.append(image)

for image_file in tqdm(noncsmoke):
    image = Image.open(noncsmoke_path + image_file)
    image = data_transform(image)
    image = np.array(image)
    # image = image / 255.0
    data.append(image)

# plt.imshow(data[-1])
# plt.show()
print('type is ', type(data))
print('length is ', len(data))
# print(data[0:1])
# print(data[-1:])

X = np.array(data)
Y = np.array(labels)
# length of X and Y
# print(len(X))
# print(len(Y))
print(X.shape, Y.shape)

# split for the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Create DataLoader
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# device configuration
device = setup_device()


# Define the modified VGG19 model
class ModifiedVGG19(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedVGG19, self).__init__()
        original_vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(original_vgg.features.children())[:14])  # Keep only first 14 layers
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_input_size = 12544
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        print("Flattened shape:", x.shape)
        x = self.classifier(x)
        print("Output shape:", x.shape)
        return x


# Initialize the model, criterion, and optimizer
num_classes = 2  # Binary classification: smoker vs non-smoker
model = ModifiedVGG19(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print('model is :', model)


# Train the model
def train_model(model, criterion, optimizer, train_loader, num_epochs=5, device='cuda'):
    model.to(device)
    model.train()
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return train_losses, train_accuracies


train_losses, train_accuracies = train_model(model, criterion, optimizer, train_loader)
print('Loss: ')
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

print('Accuracy: ')
plt.plot(train_accuracies, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Evaluate the model
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Test Accuracy (test images): %d %%' % accuracy)
    return accuracy


evaluate_model(model, test_loader, device)


# Confusion metrics
def calculate_metrics(t_positives, f_positives, t_negatives, f_negatives):
    sensitivity = t_positives / (t_positives + f_negatives)
    specificity = t_negatives / (t_negatives + f_positives)
    precision = t_positives / (t_positives + f_positives)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    return sensitivity, specificity, precision, f1_score


def get_confusion_matrix(model, data_loader):
    model.eval()
    t_positives = 0
    f_positives = 0
    t_negatives = 0
    f_negatives = 0

    with torch.no_grad():
        for images, labels, in tqdm(data_loader, desc='Calculating Confusion Matrix', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            t_positives += ((predicted == 1) & (labels == 1)).sum().item()
            f_positives += ((predicted == 1) & (labels == 0)).sum().item()
            t_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            f_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    return t_positives, t_negatives, f_positives, f_negatives


def calculate_metrics_from_confusion_matrix(model, data_loader):
    t_positives, f_positives, t_negatives, f_negatives = get_confusion_matrix(model, data_loader)
    sensitivity, specificity, precision, f1_score = calculate_metrics(t_positives, f_positives, t_negatives,
                                                                      f_negatives)
    print(f'Sensitivity (SEN) : { sensitivity * 100:.4f}')
    print(f'specificity (SPE) : {specificity * 100:.4f}')
    print(f'Precision (PRE) : {precision * 100:.4f}')
    print(f'F1-score (F1SCO) : {f1_score * 100:.4f}')


calculate_metrics_from_confusion_matrix(model, test_loader)
print('CUDA cache clean: ', torch.cuda.empty_cache())

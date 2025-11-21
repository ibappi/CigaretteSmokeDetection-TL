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


csmoke = os.listdir('./data/c_smoker')
noncsmoke = os.listdir('./data/non_c_smoker')

# Labeling the data
label_c_smoke = [1] * len(csmoke)
label_non_csmoke = [0] * len(noncsmoke)
label = label_c_smoke + label_non_csmoke

# image resize and convert
smoker_path = './data/c_smoker/'
noncsmoke_path = './data/non_c_smoker/'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

data = []

for image_file in tqdm(csmoke):
    image = Image.open(smoker_path + image_file)
    image = data_transform(image)
    data.append(image)

for image_file in tqdm(noncsmoke):
    image = Image.open(noncsmoke_path + image_file)
    image = data_transform(image)
    data.append(image)

X = torch.stack(data)
Y = torch.tensor(label)

# split for the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Create DataLoader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# load model
model = models.vgg19(pretrained=True)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the feature extractor to include only the first 3 convolutional layers
new_feature_extractor = nn.Sequential(*list(model.features.children())[:6])  # Selecting first 6 layers (3 convolutions)
model.features = new_feature_extractor

# Update num_features to match the output size of the last convolutional layer
num_features = 128  # Assuming the last convolutional layer outputs 128 features

# Update the fully connected layers to match the updated num_features
model.classifier = nn.Sequential(
    nn.Linear(num_features * 7 * 7, 4096),  # Adjust input size based on the output size of the last conv layer
    nn.ReLU(inplace=True),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 2),
    nn.Sigmoid()
)

# Move the model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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


train_loss_history, train_acc_history = train_model(model, train_loader, criterion, optimizer, num_epochs=5)
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

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = correct / total

        print(f'Test Loss: {test_loss: .4f}, Test Accuracy: {test_acc: .4f}')


evaluate_model(model, test_loader, criterion)


def calculate_metrics(t_positives, f_positives, t_negatives, f_negatives):
    sensitivity = t_positives / (t_positives + f_negatives)
    specificity = t_negatives / (t_negatives + f_positives)
    precision = t_positives / (t_positives + f_positives)

    if precision == 0 and sensitivity == 0:
        f1_score = 0
    else:
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
    print(f'Sensitivity (SEN) : {sensitivity:.4f}')
    print(f'specificity (SPE) : {specificity:.4f}')
    print(f'Precision (PRE) : {precision:.4f}')
    print(f'F1-score (F1SCO) : {f1_score:.4f}')


calculate_metrics_from_confusion_matrix(model, test_loader)

#
# # predicted system
# def define_model():
#     model = models.vgg19(pretrained=True)
#     num_features = model.classifier[6].in_features
#     model.classifier[6] = torch.nn.Linear(num_features, 2)
#     return model
#
#
# # load the model
# model = define_model()
#
# # define transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
# ])
#
#
# # predicted image
# def predicted_image(model, image, transform):
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(image)
#         _, pretedtec = output.max(1)
#     return pretedtec.item()
#
#
# # create a tinker window
# root = Tk()
# root.withdraw()
#
# # ask user for the image
# file_path = filedialog.askopenfilename(title="Choose an image file")
# if file_path:
#     image = Image.open(file_path)
#     predicted_class = predicted_image(model, image, transform)
#     if predicted_class == 0:
#         print("-------------------------------")
#         print("Predicted: User is not smoking")
#         print("-------------------------------")
#     else:
#         print("-------------------------------------------------------------------")
#         print("Predicted: User is smoking and Please do not smoke inside the car")
#         print("-------------------------------------------------------------------")

print(torch.cuda.memory_summary(device=device))
print('CUDA cache clean: ', torch.cuda.empty_cache())

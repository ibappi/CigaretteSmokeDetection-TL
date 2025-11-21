import PIL
import numpy as np
import os
from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
# csmoke = os.listdir('./data/tomato/Tomato___Bacterial_spot')
# noncsmoke = os.listdir('./data/tomato/Tomato___healthy')

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
label = label_c_smoke + label_non_csmoke

print(len(label))
print(type(label))

# image resize and convert
smoker_path = './data/c_smoker/'
noncsmoke_path = './data/non_c_smoker/'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    # transforms.ColorJitter(hue=.05, saturation=.05),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation((20, 70)),
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
Y = np.array(label)
# length of X and Y
# print(len(X))
# print(len(Y))
print(X.shape, Y.shape)

# split for the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=96)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).long()

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # batch size 16 or 32

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # batch size 16 or 32

del train_dataset, test_dataset

# device configuration
device = setup_device()

# load model
model = models.vgg19(pretrained=True)
print('Before freeze model:', model)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

print('Before modifying model:', model)
# Adding extra block to test the performance
block6 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 2),
    nn.Sigmoid()
)

model.add_module('Extra', block6)
print('After modify:- new model: ', model)

# Modify the classifier to match your problem
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

# Move the model to the device
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
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
        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # return train_loss_history
    return train_loss_history, train_acc_history


train_loss_history, train_acc_history = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

print('Loss: ')
# plt.figure(figsize=(8, 5))
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


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tp, fp, tn, fn = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        # Print evaluation metrics
        print("Evaluation after training:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1-score: {f1:.4f}")


evaluate_model(model, test_loader)


# predicted system
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


# predicted image
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

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


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


# Preprocessing
smoker_path = './data/SMV_c_smoker/'
noncsmoke_path = './data/SMV_non_c_smoker/'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

data = []
labels = []

# Load and preprocess data
for folder, label in [(smoker_path, 1), (noncsmoke_path, 0)]:
    for image_file in tqdm(os.listdir(folder)):
        image = Image.open(os.path.join(folder, image_file))
        image = data_transform(image)
        data.append(image)
        labels.append(label)

X = torch.stack(data)
Y = torch.tensor(labels)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize VGG19 model
vgg19 = models.vgg19(pretrained=True)
vgg19 = nn.Sequential(*list(vgg19.features.children())[:-1])  # Remove the last fully connected layer
num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg19.parameters(), lr=0.001)
# Move model to device
device = setup_device()
vgg19.to(device)
print(vgg19)
vgg19.classifier = nn.Identity()

svm = SVC(kernel='linear')
trin_loss_hiostory = []
for epoch in range(num_epochs):
    vgg19.train()
    running_loss = 0.0
    for images, labels in tqdm(DataLoader(list(zip(X_train, Y_train)), batch_size=32, shuffle=True), desc='Extracting images and labels'):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        features = vgg19(images)
        features = features.view(features.size(0), -1)
        outputs = svm.predict(features.cpu().numpy())
        outputs_tensors = torch.tensor(outputs, dtype=torch.float32).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epochs_loss = running_loss / len(X_train)
    trin_loss_hiostory.append(epochs_loss)
    print(f'Epoch: [{epoch + 1}/{num_epochs}], Loss: {epochs_loss:.4f}')

plt.plot(trin_loss_hiostory)
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss curve')
plt.show()

# Extract features from the VGG19 model
train_features = []
for image in tqdm(X_train, desc='Extract Train features'):
    with torch.no_grad():
        feature = vgg19.features(image.unsqueeze(0)).flatten().cpu.numpy()
        train_features.append(feature)
train_features = np.array(train_features)

# Train svm classifier
svm.fit(train_features, Y_train.cpu())

# Extract test features
test_features = []
for image in tqdm(X_test, desc='Extract test features'):
    with torch.no_grad():
        feature = vgg19.features(image.unsqueeze(0)).flatten().cpu().numpy()
        test_features.append(feature)
test_features = np.array(test_features)

# make prediction
y_pred_train = svm.predict(train_features)
y_pred_test = svm.predict(test_features)

# Calculate evaluation metrics
test_accuracy = accuracy_score(Y_test.cpu(), y_pred_test)
precision = precision_score(Y_test.cpu(), y_pred_test)
recall = recall_score(Y_test.cpu(), y_pred_test)
f1 = f1_score(Y_test.cpu(), y_pred_test)
tp, fp, tn, fn = confusion_matrix(Y_test.cpu(), y_pred_test).ravel()
specificity = tn / (tn + fp)

# Print evaluation metrics
print("Evaluation after training:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1:.4f}")


#
# # Extract features using VGG19
# def extract_features(model, dataloader):
#     model.eval()
#     features = []
#     with torch.no_grad():
#         for inputs, _ in tqdm(dataloader, desc="Extracting Features"):
#             inputs = inputs.to(device)
#             features.append(model(inputs).reshape(inputs.size(0), -1))
#     return torch.cat(features, dim=0)
#
#
# # Create DataLoaders
# train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=False)
# test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16, shuffle=False)
#
# # Extract features
# train_features = extract_features(vgg19, train_loader)
# test_features = extract_features(vgg19, test_loader)
#
# # Train SVM for 4 epochs
# num_epochs = 150
# svm_classifier = SVC(kernel='linear')
#
# train_errors = []
# test_errors = []
# train_accuracies = []
# test_accuracies = []
#
#
# for epoch in tqdm(range(num_epochs), desc='Training model'):
#     svm_classifier.fit(train_features.cpu().numpy(), Y_train.numpy())
#
#     train_predictions = []
#     test_predictions = []
#
#     # Evaluate on train set
#     train_prediction = svm_classifier.predict(train_features.cpu().numpy())
#     train_accuracy = accuracy_score(Y_train.numpy(), train_prediction)
#     print("Length of train labels:", len(Y_train.numpy()))
#     print("Length of train predictions:", len(train_prediction))
#     train_accuracies.append(train_accuracy)
#     train_error = 1 - train_accuracy
#     train_errors.append(train_error)
#     train_predictions.append(train_prediction)
#
#     # Evaluate on test set
#     test_prediction = svm_classifier.predict(test_features.cpu().numpy())
#     test_accuracy = accuracy_score(Y_test.numpy(), test_prediction)
#     test_accuracies.append(test_accuracy)
#     test_error = 1 - test_accuracy
#     test_errors.append(test_error)
#     test_predictions.append(test_prediction)
#
#     print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_error:.4f}, Test Loss: {test_error:.4f}")
#     print(f"Epoch {epoch + 1}/{num_epochs}: Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
#
#
# # Plotting
# plt.figure(figsize=(7, 6))
#
# epochs = list(range(1, num_epochs + 1))  # Correcting the epochs list
#
# plt.plot(epochs, train_errors, label='Train Error')  # marker='o'
# plt.plot(epochs, test_errors, label='Test Error', marker='o')
# plt.title('Training and Test Error')
# plt.xlabel('Epochs')
# plt.ylabel('Error')
# # plt.xticks(epochs)
#
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
# plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
# plt.title('Training and Test Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# # plt.xticks(epochs)
# plt.legend()
# plt.grid(True)
# plt.show()
#
# train_predictions = np.concatenate(train_predictions)
# test_predictions = np.concatenate(test_predictions)
#
# # Ensure the lengths match
# assert len(Y_train.numpy()) == len(train_predictions)
# assert len(Y_test.numpy()) == len(test_predictions)
#
# confusion_mat_train = confusion_matrix(Y_train.numpy(), train_predictions)
# confusion_mat_test = confusion_matrix(Y_test.numpy(), test_predictions)
#
# # Combine predictions and labels from train and test sets
# combined_predictions = np.concatenate((train_predictions, test_predictions))
# combined_labels = np.concatenate((Y_train.numpy(), Y_test.numpy()))
#
# # Calculate confusion matrix
# confusion_mat_combined = confusion_matrix(combined_labels, combined_predictions)
#
# # Calculate precision, recall, and specificity
# precision = confusion_mat_combined[1, 1] / (confusion_mat_combined[1, 1] + confusion_mat_combined[0, 1])
# sensitivity = confusion_mat_combined[1, 1] / (confusion_mat_combined[1, 1] + confusion_mat_combined[1, 0])
# specificity = confusion_mat_combined[0, 0] / (confusion_mat_combined[0, 0] + confusion_mat_combined[0, 1])
# f1score = f1_score(combined_labels, combined_predictions)
#
# # Print evaluation metrics
# print("Evaluation after all epochs:")
# print(f"Precision: {precision:.4f}")
# print(f"Sensitivity: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"F1-score: {f1score:.4f}")

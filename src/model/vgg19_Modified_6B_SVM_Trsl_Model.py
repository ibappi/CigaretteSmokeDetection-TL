import os
import time
import PIL
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib as mpl
import thop
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
smoker_path = './data/SMV_c_smoker/'
noncsmoke_path = './data/SMV_non_c_smoker/'

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
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

X = torch.stack(data).to(device)
Y = torch.tensor(labels).to(device)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize VGG19 model
vgg19 = models.vgg19(pretrained=True).to(device)

# Freeze pretrained layers
for param in vgg19.parameters():
    param.requires_grad = False

# Define the additional convolutional layers
block6 = nn.Sequential(
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=2, padding=1),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, padding=1),
).to(device)

# Adjust the convolutional layer in block6 to have the same output size
# block6[-1] = nn.Conv2d(512, 512, kernel_size=3, padding=1)

# Replace the last layers of the pretrained model
vgg19.classifier = nn.Sequential(
    nn.Linear(512 * 3 * 3, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 1000),  # Original number of output classes
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(1000, 2),  # Output layer for binary classification
).to(device)

# Append block6 to the feature extraction part
vgg19.features.add_module('block6', block6)


# # ---------------------------------------------
# # Define a hook function to print input and output shapes
def print_shapes(module, input, output):
    print(f"Module: {module.__class__.__name__}")
    print(f"  Input shape: {input[0].shape}")
    print(f"  Output shape: {output.shape}")


# Register the hook function to each layer
for name, module in vgg19.named_modules():
    module.register_forward_hook(print_shapes)
# # ---------------------------------------------
# Fine-tune the entire model for multiple epochs

num_epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg19.parameters(), lr=0.001)
vgg19.to(device)

# Train the model
train_loss_history = []
for epoch in range(num_epochs):
    vgg19.train()
    running_loss = 0.0
    for images, labels in tqdm(DataLoader(list(zip(X_train, Y_train)), batch_size=32, shuffle=True), desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        features = vgg19.features(images)
        features = features.view(features.size(0), -1)
        outputs = vgg19.classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(X_train)
    train_loss_history.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Plot train loss values
import matplotlib.pyplot as plt
plt.plot(train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Curve')
plt.show()

# Extract features from the modified VGG19 model
train_features = []
for image in tqdm(X_train, desc='Extract train features'):
    with torch.no_grad():
        feature = vgg19.features(image.unsqueeze(0)).flatten().cpu().numpy()
        train_features.append(feature)
train_features = np.array(train_features)

# Train SVM classifier with linear kernel for making hyperplane in higher dimensions
svm = SVC(kernel='linear')
svm.fit(train_features, Y_train.cpu())

# Extract features from test set
test_features = []
for image in tqdm(X_test, desc='Extract test features'):
    with torch.no_grad():
        feature = vgg19.features(image.unsqueeze(0)).flatten().cpu().numpy()
        test_features.append(feature)
test_features = np.array(test_features)

# Make predictions
Y_pred_train = svm.predict(train_features)
start_time = time.time()
Y_pred_test = svm.predict(test_features)
end_time = time.time()
inference_time = (end_time - start_time) * 1000

# Calculate evaluation metrics
test_accuracy = accuracy_score(Y_test.cpu(), Y_pred_test)
precision = precision_score(Y_test.cpu(), Y_pred_test)
recall = recall_score(Y_test.cpu(), Y_pred_test)
f1 = f1_score(Y_test.cpu(), Y_pred_test)
tp, fp, tn, fn = confusion_matrix(Y_test.cpu(), Y_pred_test).ravel()
specificity = tn / (tn + fp)

# Print evaluation metrics
print("Evaluation after training:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Inference time: {inference_time:.4f}")


# Calculate the GFLOPs
# Define input and run THOP to get model complexity
inputs = X_train[0].unsqueeze(0).to(device)
flops, params = thop.profile(vgg19, inputs=(inputs,), verbose=False)

# Convert FLOPs to GFLOPs and print results
gflops = flops / 1e9
print(f"Total FLOPs: {gflops:.4f} GFLOPs")

# Clean CUDA cache
print(torch.cuda.memory_summary(device=device))
print('CUDA cache clean: ', torch.cuda.empty_cache())
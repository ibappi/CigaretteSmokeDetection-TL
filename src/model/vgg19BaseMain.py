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

# Move model to device
device = setup_device()
vgg19.to(device)
print(vgg19)


# Extract features using VGG19
def extract_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            features.append(model(inputs).reshape(inputs.size(0), -1))
    return torch.cat(features, dim=0)


# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16, shuffle=False)


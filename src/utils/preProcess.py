# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import models, transforms
# from tqdm import tqdm
#
# csmoke = os.listdir('./data/c_smoker')
# noncsmoke = os.listdir('./data/non_c_smoker')
#
# print(csmoke[0:5])
# print(noncsmoke[0:5])
#
# # img = mpimg.imread('./data/c_smoker/' + csmoke[1])
# # plt.imshow(img)
# # plt.show()
#
# # length of folder
# print(len(csmoke))
# print(len(noncsmoke))
# #
# # print(type(csmoke))
# # print(type(noncsmoke))
#
# # Labeling the data
# label_c_smoke = [1] * len(csmoke)
# label_non_csmoke = [0] * len(noncsmoke)
# label = label_c_smoke + label_non_csmoke
# # print(label)
# print(len(label))
# print(type(label))
#
# # image resize and convert
#
# smoker_path = './data/c_smoker/'
# noncsmoke_path = './data/non_c_smoker/'
#
# data_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
# ])
# print('type of data', type(data_transform))
#
# data = []
#
# for image_file in tqdm(csmoke):
#     image = Image.open(smoker_path + image_file)
#     image = data_transform(image)
#     image = np.array(image)
#     # image = image / 255.0
#     data.append(image)
#
# for image_file in tqdm(noncsmoke):
#     image = Image.open(noncsmoke_path + image_file)
#     image = data_transform(image)
#     image = np.array(image)
#     # image = image / 255.0
#     data.append(image)
#
# # plt.imshow(data[-1])
# # plt.show()
# print('type is ', type(data))
# print('length is ', len(data))
# print(data[0:1])
# # print(data[-1:])
#
# X = np.array(data)
# Y = np.array(label)
# # length of X and Y
# # print(len(X))
# # print(len(Y))
# print(X.shape, Y.shape)
# print('X is ', X)
# print('Y is ', Y)
# # split fot the , model
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
#
# # Convert data to PyTorch tensors
# X_train_tensor = torch.from_numpy(X_train).float()
# Y_train_tensor = torch.from_numpy(Y_train).long()
# X_test_tensor = torch.from_numpy(X_test).float()
# Y_test_tensor = torch.from_numpy(Y_test).long()
#
# # Create DataLoader
# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # batch size 16 or 32
#
# test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # batch size 16 or 32
#
# del train_dataset, test_dataset
#
# # old model
# # Load VGG19 model
# vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
# # Freeze all the layers except the final classifier
# # for param in vgg19.parameters():
# #     param.requires_grad = False
#
# # Modify the model to use only the feature extraction part
# vgg19_feature = vgg19.features
#
# # Remove the last layer (classifier) of VGG19
# vgg19_feature = vgg19_feature[:-1]
#
# num_features = 512  # Assuming the last convolutional layer has 512 output channels
#
# vgg19_feature.add_module('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
#
# classifier = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(num_features, 512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(512, 2)
# )
# model = nn.Sequential(vgg19_feature, classifier)
#
# # Use CUDA with GPUs
# use_cuda = torch.cuda.is_available()
#
# if use_cuda:
#     num_gpus = torch.cuda.device_count()
#     if num_gpus >= 1:
#         # Use both GPUs
#         device_ids = [i for i in range(num_gpus)]
#         print(f'Running with {num_gpus} GPU: {device_ids}')
#         print(f'Running with {num_gpus} GPUs')
#         model = nn.DataParallel(model, device_ids=device_ids)
#         model = model.to(device_ids[0])
#
#     else:
#         # Use single GPU
#         device = torch.device('cuda:0')
#         print(f'Running with {num_gpus} GPU')
#         model = model.to(device)
# else:
#     device = torch.device('cpu')
#     print('Running with CPU')
#     model = model.to(device)
#
# # ---------------------
# print(model)
# print("Number of parameters:", sum(p.numel() for p in model.parameters()))
#
# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Initialize scaler for mixed-precision training
# scaler = GradScaler()
#
# # Training loop
# train_loss_history = []
# train_accuracy_history = []
# num_epochs = 3
# accumulation_steps = 3  # Increase or decrease based on your GPU memory
# for epoch in range(num_epochs):
#     vgg19_feature.train()  # Set the feature extractor to training mode
#     total_loss = 0
#     correct_predictions = 0
#     total_samples = 0
#     accumulation_counter = 0
#
#     for inputs, labels in tqdm(train_loader):
#         inputs, labels = inputs.to(device_ids[0]), labels.to(device_ids[0])  # Move inputs and labels to the first GPU
#
#         optimizer.zero_grad()
#
#         # Use autocast to enable mixed-precision training
#         with autocast():
#             # Pass inputs through the feature extractor part of the VGG model
#             features = vgg19_feature(inputs)
#
#             # Modify the shape to match the expected format
#             features = features.view(features.size(0), -1)
#
#             # Access the classifier within the DataParallel wrapper
#             classifier = model.module[1]
#
#             # Pass the features through the modified classifier
#             outputs = classifier(features)
#
#             loss = criterion(outputs, labels)
#             print('Outputs: ', outputs.shape)
#
#         # Use scaler to scale loss and perform backpropagation
#         scaler.scale(loss).backward()
#         accumulation_counter += 1
#
#         if accumulation_counter == accumulation_steps:
#             # Perform optimizer step only after accumulating gradients for accumulation_steps batches
#             scaler.step(optimizer)
#             scaler.update()
#             accumulation_counter = 0
#
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         correct_predictions += (predicted == labels).sum().item()
#         total_samples += labels.size(0)
#
#     # Calculate average loss and accuracy for the epoch
#     avg_loss = total_loss / len(train_loader)
#     avg_accuracy = correct_predictions / total_samples
#
#     # Append values for plotting
#     train_loss_history.append(avg_loss)
#     train_accuracy_history.append(avg_accuracy)
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy * 100:.2f}%')
#     print(torch.cuda.memory_summary(device=device_ids[0]))
#
# # Plotting the loss and accuracy
# plt.figure(figsize=(10, 4))
#
# # Plot Loss
# plt.subplot(1, 2, 1)
# plt.plot(train_loss_history, label='Training Loss')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# # Plot Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracy_history, label='Training Accuracy', color='orange')
# plt.title('Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # Deleting the cache
# torch.cuda.empty_cache()
#
# # Evaluation on the test set
# print('Start evaluate!')
# model.eval()
# criterion = nn.MSELoss()
# losses = 0
# with torch.no_grad():
#     with torch.cuda.amp.autocast():
#         for data in tqdm(test_loader):
#             test_X, test_y = data
#             test_X, test_y = test_X.to(device_ids[0]), test_y.to(device_ids[0])
#             preds = model(test_X)
#             loss = criterion(preds, test_y)
#             losses += loss.item() / len(test_loader)
#
#         print(losses)
#
#         # X_test_tensor = X_test_tensor.to(device_ids[0])  # Move test data to the first GPU
#         # Y_pred = vgg19(X_test_tensor)
#         # _, predicted = torch.max(Y_pred, 1)
#
# # Print GPU memory usage summary after evaluation
# # print(torch.cuda.memory_summary(device=device_ids[0]))
# #
# # # Free up GPU memory after evaluation
# # del Y_pred, predicted
# # torch.cuda.empty_cache()
# #
# # accuracy = (predicted == Y_test_tensor).sum().item() / Y_test_tensor.size(0)
# # print(f'Test Accuracy: {accuracy * 100:.2f}%')
#

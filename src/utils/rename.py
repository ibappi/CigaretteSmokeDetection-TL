import os
import cv2

# folder = './data/c_smoker/'
folder = './data/non_c_smoker/'

count = 1

for file_name in os.listdir(folder):
    source = folder + file_name
    destination = folder + "noncsmoker_" + str(count) + ".jpg"
    os.rename(source, destination)
    count += 1

print('all files renamed')


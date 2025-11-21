"""
  ____             _   _
 / ___|  ___  _ __| |_(_)_ __   __ _
 \___ \ / _ \| '__| __| | '_ \ / _` |
  ___) | (_) | |  | |_| | | | | (_| |
 |____/ \___/|_|   \__|_|_| |_|\__, |
                               |___/
This script uses the customtkinter module to create a simple image viewer GUI that helps sort the images into 2 classes to create a proper dataset / to clean the data.
"""

import customtkinter as ctk  # Importing customtkinter module as ctk
from tkinter import *  # Importing everything from the tkinter module
from PIL import Image  # Importing the Image module from the PIL library
import os  # to load and iterate through folders/images
import glob  # to get the image count

ctk.set_default_color_theme("dark-blue")  # Set the default color theme for customtkinter: "dark-blue"
root = ctk.CTk()  # Creating a custom tkinter instance
root.title("Data Sorter")  # Setting the title of the main window to "Image Viewer"

folder_dir = None  # where the path will be stored
image = None  # where the image name will be stored
number_of_files = 0  # saves the number of images in the folder
counter = 0  # counts the current image the user is on
already_done = []  # saves the names of the images that are already sorted into one of the two folders from a previous run


def good_image(event=None):  # if green button / right arrow is pressed
    global next, folder_dir, image, PIL_Image  # global vars
    if not folder_dir:  # to prevent errors if pressed before image is loaded
        status.configure(text="No Folder selected yet - Try again!")  # status message to inform user
        return  # exit before error
    PIL_Image.save(f"{folder_dir}/good/{image}")  # save image to good folder
    next.set(True)  # trigger next image


def bad_image(event=None):  # if red button / left arrow is pressed
    global next, folder_dir, image, PIL_Image  # global vars
    if not folder_dir:  # to prevent errors if pressed before image is loaded
        status.configure(text="No Folder selected yet - Try again!")  # status message to inform user
        return  # exit before error
    PIL_Image.save(f"{folder_dir}/bad/{image}")  # save image to good folder
    next.set(True)  # trigger next image


def quit_window():  # close window
    exit_loop.set(True)  # to stop the image loop
    next.set(True)  # one last step into the loop
    root.quit()  # close window


def select_folder():  # opens dialog window to select folder
    global my_image, label, status, next, root, exit_loop, folder_dir, image, number_of_files, already_done, counter  # global vars

    status.configure(text="Select Image from Dialog Window")  # status message

    folder_dir = ctk.filedialog.askdirectory(  # folder dialog window
        title='Select a Folder',
        initialdir='/',
    )
    if not folder_dir:  # check if folder selected to prevent error
        status.configure(text="No Folder selected - Try again!")  # status message
        return  # exit before error code would run

    status.configure(text="Selecting Folder...")  # status message
    status.configure(text="Folder Selected, Images will be Loaded!")  # status message

    number_of_files = len(glob.glob(f"{folder_dir}/*.jpg"))  # counts how many images are in folder

    if os.path.exists(f"{folder_dir}/good") and os.path.exists(
            f"{folder_dir}/bad"):  # check if folders already exist (previous runs)
        for image in os.listdir(f"{folder_dir}/good"):  # all imaegs ingood folder
            if (image.endswith(".jpg")):  # if legal
                already_done.append(image)  # append name to list of previous
                counter += 1  # increase counter
        for image in os.listdir(f"{folder_dir}/bad"):  # all imaegs ingood folder
            if (image.endswith(".jpg")):  # if legal
                already_done.append(image)  # append name to list of previous
                counter += 1  # increase counter

    if not os.path.exists(f"{folder_dir}/good"):  # if folder is not there
        os.mkdir(f"{folder_dir}/good")  # make it
    if not os.path.exists(f"{folder_dir}/bad"):  # if folder is not there
        os.mkdir(f"{folder_dir}/bad")  # make it

    for image in os.listdir(folder_dir):  # iterate through all files in folder
        if (image.endswith(".jpg")):  # if image
            if image not in already_done:  # if image not already in folders from previous run
                process_image()  # process
        if exit_loop.get():  # if exit button pressed
            break  # break out of loop


def process_image():  # display image and allow classification
    global my_image, label, status, folder_dir, image, PIL_Image, counter, number_of_files  # global vars
    counter += 1  # increase counter
    status.configure(
        text=f"Image {counter}/{number_of_files} loaded!\n Use Buttons next to image or 'Arrow Keys' to sort.")  # status message

    PIL_Image = Image.open(f"{folder_dir}/{image}")  # Opening the selected image file using PIL
    my_image = ctk.CTkImage(dark_image=PIL_Image, size=(666, 666))  # Creating a CTkImage object from the PIL image
    label.configure(image=my_image, text="")  # Configuring the label to display the loaded image

    status.wait_variable(next)  # wait for button press to keep looping
    next.set(False)  # set to false right after lasy trigger


"""
This code is creating a GUI (Graphical User Interface) for an image viewer application using the
customtkinter module.
"""

next = ctk.BooleanVar()  # variable that triggers loop progession
next.set("False")  # default false
exit_loop = ctk.BooleanVar()  # variable to trigger exit condition
exit_loop.set("False")  # default false

label = ctk.CTkLabel(root, width=666, height=666, text="Select Image")  # Creating a label widget for displaying images
label.grid(row=1, column=1, columnspan=4)  # Placing the label in the grid layout

button_good = ctk.CTkButton(root, command=good_image, text=">", fg_color="#239E8B", hover_color="#10473E",
                            corner_radius=50, width=10, text_color="#2E2E2E")  # Good, Green
button_bad = ctk.CTkButton(root, command=bad_image, text="<", fg_color="#9E2B2B", hover_color="#641B1B",
                           corner_radius=50, width=10, text_color="#2E2E2E")  # Bad, Red

button_bad.grid(row=1, column=0, pady=3)  # Left
button_good.grid(row=1, column=6, pady=3)  # Right

root.bind('<Left>', bad_image)  # Left Arrow key, Bad button
root.bind('<Right>', good_image)  # Right Arrow key, Good button

status = ctk.CTkLabel(root, text="Select Image")  # Creating a label widget for displaying status messages
status.grid(row=2, column=1, columnspan=4)  # Placing the status message label in the grid layout

# Creating buttons for loading, exiting, and running the model
button_load = ctk.CTkButton(root, command=select_folder, text="Load", fg_color="#2B719E",
                            hover_color="#1B4864")  # Load, Blue
button_exit = ctk.CTkButton(root, text="Exit", fg_color="#9E2B2B", hover_color="#641B1B",
                            command=quit_window)  # Exit, Red

# Placing the buttons in the grid layout
button_exit.grid(row=5, column=2, pady=3)  # Left
button_load.grid(row=5, column=3, pady=3)  # Center Left

root.mainloop()  # Running the main event loop to display the GUI
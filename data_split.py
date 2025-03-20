import os
import random
from shutil import copyfile

# Paths to input images and labels folders
input_images_folder = "D:/HeadCount_FaceAttendance_Project/Student_Data/Students/Images"
input_labels_folder = "D:/HeadCount_FaceAttendance_Project/Student_Data/Students/Labels"

# Output folders for train and valid sets
output_train_folder = "D:/HeadCount_FaceAttendance_Project/Student_Data/Students/train"
output_valid_folder = "D:/HeadCount_FaceAttendance_Project/Student_Data/Students/valid"

# Create output folders if they don't exist
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_valid_folder, exist_ok=True)

# List files in input folders
images = os.listdir(input_images_folder)
random.shuffle(images)  # Shuffle for random split

# Split into train and valid sets (e.g., 80% train, 20% valid)
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
valid_images = images[split_idx:]

def copy_files(images_list, input_folder, output_folder):
    for image in images_list:
        image_name = os.path.splitext(image)[0]
        label_file = f"{image_name}.txt"
        # Create subfolders for images and labels
        os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)
        copyfile(os.path.join(input_folder, image), os.path.join(output_folder, 'images', image))
        copyfile(os.path.join(input_labels_folder, label_file), os.path.join(output_folder, 'labels', label_file))

# Copy images and labels to train folder
copy_files(train_images, input_images_folder, output_train_folder)

# Copy images and labels to valid folder
copy_files(valid_images, input_images_folder, output_valid_folder)
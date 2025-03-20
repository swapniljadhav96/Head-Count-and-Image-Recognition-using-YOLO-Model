from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
import os

# Directory containing folders, each with images
input_dir = 'updated_students'
output_dir = 'Augmented_Faces'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=0,  # No rotation
    width_shift_range=0.1,  # Random horizontal shift by 10%
    height_shift_range=0.1,  # Random vertical shift by 10%
    shear_range=0,  # No shearing
    zoom_range=0.1,  # Random zoom by 10%
    horizontal_flip=False,  # No horizontal flip
    vertical_flip=False,  # No vertical flip
    fill_mode='nearest'  # Fill mode for pixels outside the boundaries
)

# Loop through each folder in the input directory
for label in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, label)
    if os.path.isdir(folder_path):
        # Create the output directory for the label if it doesn't exist
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        # Find image files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpeg') or filename.endswith('.png'):
                # Load the image
                image_path = os.path.join(folder_path, filename)

                # Generate augmented images
                img = load_img(image_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=output_label_dir, save_prefix='aug', save_format='jpeg'):
                    i += 1
                    if i >= 10:  # Generate 10 augmented images per original image
                        break

print("Data augmentation completed.")
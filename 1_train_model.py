import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils import class_weight



# Paths to your folders
positive_dir = "C:\\Users\\jessj\\OneDrive\\Documents\\glaucoma\\archive\\Fundus_Train_Val_Data\\Fundus_Scanes_Sorted\\Train\\Glaucoma_Positive"
negative_dir = "C:\\Users\\jessj\\OneDrive\\Documents\\glaucoma\\archive\\Fundus_Train_Val_Data\\Fundus_Scanes_Sorted\\Train\\Glaucoma_Negative"

# Initialize lists to store images and labels
images = []
labels = []

# Function to load images from a folder and assign labels
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))  # Resize to match model input size
            image = img_to_array(image)  # Convert image to array
            image = image / 255.0  # Normalize the image
            images.append(image)
            labels.append(label)

# Load positive images (label 1 for glaucoma)
load_images_from_folder(positive_dir, 1)

# Load negative images (label 0 for normal)
load_images_from_folder(negative_dir, 0)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training images: {X_train.shape}")
print(f"Testing images: {X_test.shape}")

# Build the CNN model
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer with max pooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected layer with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification (glaucoma vs. normal)

 
 
# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Train the model with class weights
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)
# Save the trained model
model.save('glaucoma_model.h5')

print("Model training completed and saved as 'glaucoma_model.h5'")

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping  
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight  
import seaborn as sns


def preprocess_image(image_path, target_size=(640, 480)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size) 
    image = image / 255.0  
    image = img_to_array(image)  
    return image


csv_file = r"C:\Users\dej72\OneDrive\Desktop\Projects\Project2\allimages\all\labled_updated.csv"  
df = pd.read_csv(csv_file)


images = []
labels = []


for index, row in df.iterrows():
    image_path = row['full_image_path']  
    label = row['label']  
    
    # Preprocess the image
    image = preprocess_image(image_path, target_size=(640, 480))
    images.append(image)
    

    if label == 'left':
        labels.append(0)
    elif label == 'right':
        labels.append(1)
    elif label == 'straight':
        labels.append(2)
    elif label == 'no lane':
        labels.append(3)

# Convert the images and labels lists into numpy arrays
X = np.array(images)
y = np.array(labels)

# Split the dataset into training and validation 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',  
    classes=np.unique(y),     
    y=y                       
)


class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights:", class_weight_dict)

# Visualize a few images from the dataset
def visualize_images(images, labels, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

visualize_images(X, y, num_images=5)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(640, 480, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(4, activation='softmax'))  # 4 output classes (left, right, straight, no lane)

# Compiling model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',         
    patience=5,                 
    restore_best_weights=True   
)

# early stopping and class weights
history = model.fit(
    X_train, y_train,
    epochs=40,                   
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],   
    class_weight=class_weight_dict 
)

# saved_model_path = r"C:\Users\dej72\OneDrive\Desktop\Projects\Project2\allimages\all\lanedetectionlearningrate2.keras"
# model.save(saved_model_path)  

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Visualize predictions
def visualize_predictions(images, true_labels, model, num_images=5):
    plt.figure(figsize=(10, 10))
    predictions = model.predict(images[:num_images])

    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        true_label = true_labels[i]
        pred_label = np.argmax(predictions[i])  # The predicted label

        # Map label indices back to class names
        class_names = ['left', 'right', 'straight', 'no lane']
        true_label_name = class_names[true_label]
        pred_label_name = class_names[pred_label]

        plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
        plt.axis('off')
    plt.show()

visualize_predictions(X_val, y_val, model, num_images=5)

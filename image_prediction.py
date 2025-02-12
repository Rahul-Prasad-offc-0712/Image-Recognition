import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Step 1: Load dataset (for multiple categories)
def load_dataset(dataset_path, img_size=(64, 64)):
    images, labels = [], []
    categories = ["Apple","Bean","Big Truck","Brinjal","Cabbage","Carrot","City Car","Grapes","Laptop","Multi Purpose Vehicle","Orange","Pineapple","Sedan","Sport utility Vehicle","Strawberry","Truck","Van"]  # List of your categories

    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)

        # Check if the folder exists
        if os.path.exists(category_path):
            files = os.listdir(category_path)
            if not files:
                print(f"No files found in {category_path}")
            else:
                # Loop through all files in the folder and load images
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    if os.path.isfile(image_path):  # Ensure it's a file, not a folder
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = cv2.resize(image, img_size)
                            images.append(image)
                            labels.append(label)  # Label images with the corresponding category's index
        else:
            print(f"Error: The folder {category_path} does not exist.")
    
    return np.array(images), np.array(labels)

# Step 2: Define CNN model (for multi-class classification)
def create_model(num_classes=4):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Change to softmax for multi-class classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train and save the model
def train_and_save_model(dataset_path, model_path="image_classifier.h5"):
    images, labels = load_dataset(dataset_path)
    
    if len(images) == 0:
        print("No images found in the dataset. Please check the dataset path.")
        return
    
    images = images / 255.0  # Normalize images

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model(num_classes=len(np.unique(labels)))  # Dynamically set number of classes
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=16)
    
    model.save(model_path)
    print("Model trained and saved!")

# Step 4: Load model and predict image (multi-class)
def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    categories = ["Laptop", "Vehicle", "Fruit", "Vegetable"]  # List of your categories

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    image = cv2.resize(image, (64, 64)) / 255.0
    image = np.expand_dims(image, axis=0)  # Reshape for model input

    prediction = model.predict(image)

    # Get the class index with the highest probability
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index] * 100  # Confidence of the predicted class
    label = categories[class_index]  # Class label

    # Display Image with Prediction
    output_image = cv2.imread(image_path)
    cv2.putText(output_image, f"{label}: {confidence:.2f}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Predicted Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main Execution
dataset_path = "E:\\Python-Image\\Dataset"  # Change to your dataset folder
model_path = "image_classifier.h5"

# Step 2: Train model
train_and_save_model(dataset_path, model_path)

# Step 3: Predict new image
test_image_path = "E:\\Python-Image\\Testing Dataset\\Apple2.jpg"  # Change to actual test image path
predict_image(model_path, test_image_path)

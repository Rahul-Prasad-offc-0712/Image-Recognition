import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
def load_dataset(dataset_path, img_size=(64, 64)):
    print(f"Loading dataset from {dataset_path}...")
    images, labels = [], []
    categories = ["Apple","Bean","Big Truck","Brinjal","Cabbage","Carrot","City Car",
                  "Grapes","Laptop","Multi Purpose Vehicle","Orange","Pineapple",
                  "Sedan","Sport utility Vehicle","Strawberry","Truck","Van"]
    
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, img_size)
                        images.append(image)
                        labels.append(label)
        else:
            print(f"Warning: Folder {category_path} does not exist.")
    
    print(f"Dataset loaded. Total images: {len(images)}")
    return np.array(images), np.array(labels)

# Step 2: Define CNN model
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
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

    print("Creating model...")
    model = create_model(num_classes=len(set(labels)))  # Use actual number of categories
    print("Model created. Starting training...")
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
    
    model.save(model_path)
    print("Model training completed and saved at", model_path)

# Step 4: Load model and predict image
# Step 4: Load model and predict image
def predict_image(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}.")

    categories = ["Apple", "Bean", "Big Truck", "Brinjal", "Cabbage", "Carrot", "City Car",
                  "Grapes", "Laptop", "Multi Purpose Vehicle", "Orange", "Pineapple",
                  "Sedan", "Sport utility Vehicle", "Strawberry", "Truck", "Van"]

    while True:
        image_path = input("\nEnter image path (or type 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break

        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found! Try again.")
            continue

        # Prepare image for model prediction
        image_resized = cv2.resize(image, (64, 64)) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)  # Reshape for model input

        # Make prediction
        prediction = model.predict(image_resized)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index] * 100
        label = categories[class_index]

        print(f"Predicted Category: {label}, Confidence: {confidence:.2f}%")

        # Annotate the image with the prediction result
        annotated_image = image.copy()  # Copy image to draw text on it
        text = f"{label} ({confidence:.2f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 255, 0)  # Green color
        text_position = (10, 30)  # Top-left corner

        # Put text on the image
        cv2.putText(annotated_image, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Display the annotated image
        cv2.imshow("Predicted Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Main Execution
dataset_path = "E:\\Python-Image\\Dataset"  # Update with actual dataset path
model_path = "image_classifier.h5"

# Train model
train_and_save_model(dataset_path, model_path)

# Predict new images in a loop
predict_image(model_path)